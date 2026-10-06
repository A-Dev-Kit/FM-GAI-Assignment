# ADDED: new file, not part of the upstream SR3 codebase.
"""Device selection and GPU memory/timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Turn ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:N"`` into a usable device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {name!r} requested but CUDA is not available.")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: torch.device) -> float | None:
    """Peak allocated GPU memory since the last reset, or ``None`` on CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


@contextmanager
def timed(device: torch.device) -> Iterator[dict[str, float]]:
    """Measure wall-clock seconds of the block, synchronising CUDA on entry and exit."""
    result: dict[str, float] = {}
    synchronize(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result["seconds"] = time.perf_counter() - start
