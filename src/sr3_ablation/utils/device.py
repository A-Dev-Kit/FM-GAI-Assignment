# ADDED: new file, not part of the upstream SR3 codebase.
"""Device selection and GPU memory/timing helpers."""  # ADDED

from __future__ import annotations  # ADDED

import time  # ADDED
from collections.abc import Iterator  # ADDED
from contextlib import contextmanager  # ADDED

import torch  # ADDED


def resolve_device(name: str = "auto") -> torch.device:  # ADDED
    """Turn ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:N"`` into a usable device."""  # ADDED
    if name == "auto":  # ADDED
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ADDED
    device = torch.device(name)  # ADDED
    if device.type == "cuda" and not torch.cuda.is_available():  # ADDED
        raise RuntimeError(f"Device {name!r} requested but CUDA is not available.")  # ADDED
    return device  # ADDED


def synchronize(device: torch.device) -> None:  # ADDED
    """Wait for queued CUDA work so timings are accurate (no-op on CPU)."""  # ADDED
    if device.type == "cuda":  # ADDED
        torch.cuda.synchronize(device)  # ADDED


def reset_peak_memory(device: torch.device) -> None:  # ADDED
    """Reset the peak GPU memory counter (no-op on CPU)."""  # ADDED
    if device.type == "cuda":  # ADDED
        torch.cuda.reset_peak_memory_stats(device)  # ADDED


def peak_memory_mib(device: torch.device) -> float | None:  # ADDED
    """Peak allocated GPU memory since the last reset, or ``None`` on CPU."""  # ADDED
    if device.type != "cuda":  # ADDED
        return None  # ADDED
    return torch.cuda.max_memory_allocated(device) / 2**20  # ADDED


@contextmanager  # ADDED
def timed(device: torch.device) -> Iterator[dict[str, float]]:  # ADDED
    """Measure wall-clock seconds of the block, synchronising CUDA on entry and exit."""  # ADDED
    result: dict[str, float] = {}  # ADDED
    synchronize(device)  # ADDED
    start = time.perf_counter()  # ADDED
    try:  # ADDED
        yield result  # ADDED
    finally:  # ADDED
        synchronize(device)  # ADDED
        result["seconds"] = time.perf_counter() - start  # ADDED
