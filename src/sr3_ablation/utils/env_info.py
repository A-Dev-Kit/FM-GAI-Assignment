# ADDED: new file, not part of the upstream SR3 codebase.
"""Record the software and hardware environment of a run (for the README and report)."""  # ADDED

from __future__ import annotations  # ADDED

import platform  # ADDED
import subprocess  # ADDED
import sys  # ADDED
from importlib import metadata  # ADDED
from typing import Any  # ADDED

import torch  # ADDED

from sr3_ablation.paths import PROJECT_ROOT  # ADDED

TRACKED_PACKAGES = ("torch", "torchvision", "numpy", "pillow", "matplotlib", "gdown", "lpips")  # ADDED


def collect_environment(device: torch.device) -> dict[str, Any]:  # ADDED
    """Python, platform, package versions, CUDA, device and git commit."""  # ADDED
    info: dict[str, Any] = {  # ADDED
        "python": sys.version.split()[0],  # ADDED
        "platform": platform.platform(),  # ADDED
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},  # ADDED
        "cuda_available": torch.cuda.is_available(),  # ADDED
        "torch_cuda": torch.version.cuda,  # ADDED
        "device": str(device),  # ADDED
        "git_commit": _git_commit(),  # ADDED
    }  # ADDED
    if device.type == "cuda":  # ADDED
        properties = torch.cuda.get_device_properties(device)  # ADDED
        info["gpu"] = {  # ADDED
            "name": properties.name,  # ADDED
            "total_memory_mib": round(properties.total_memory / 2**20),  # ADDED
            "capability": f"{properties.major}.{properties.minor}",  # ADDED
        }  # ADDED
    return info  # ADDED


def _version(package: str) -> str | None:  # ADDED
    try:  # ADDED
        return metadata.version(package)  # ADDED
    except metadata.PackageNotFoundError:  # ADDED
        return None  # ADDED


def _git_commit() -> str | None:  # ADDED
    try:  # ADDED
        completed = subprocess.run(  # ADDED
            ["git", "rev-parse", "HEAD"],  # ADDED
            cwd=PROJECT_ROOT,  # ADDED
            capture_output=True,  # ADDED
            text=True,  # ADDED
            check=True,  # ADDED
            timeout=10,  # ADDED
        )  # ADDED
    except (OSError, subprocess.SubprocessError):  # ADDED
        return None  # ADDED
    return completed.stdout.strip() or None  # ADDED
