# ADDED: new file, not part of the upstream SR3 codebase.
"""Check that the schedule loaded inside the model equals the independently computed one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sr3_ablation.schedule.noise_schedule import NoiseSchedule

FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-7
FLOAT64_RTOL = 1e-10


class ScheduleMismatchError(RuntimeError):
    """The model is not using the schedule the run declares."""


@dataclass(frozen=True)
class ScheduleBuffers:
    """Schedule buffers read back from a model.

    ``betas`` and ``alphas_cumprod`` are float32 tensors in SR3; ``sqrt_alphas_cumprod_prev`` is
    a float64 NumPy array that drives the continuous noise level fed to the UNet.
    """

    betas: np.ndarray
    alphas_cumprod: np.ndarray
    sqrt_alphas_cumprod_prev: np.ndarray


def verify_schedule(observed: ScheduleBuffers, expected: NoiseSchedule) -> dict[str, Any]:
    """Compare ``observed`` model buffers with ``expected`` and return the evidence record.

    Raises:
        ScheduleMismatchError: if any buffer differs beyond floating-point tolerance.
    """
    checks = {
        "betas": _compare(observed.betas, expected.betas, FLOAT32_RTOL, 0.0),
        "alphas_cumprod": _compare(
            observed.alphas_cumprod, expected.alphas_cumprod, FLOAT32_RTOL, FLOAT32_ATOL
        ),
        "sqrt_alphas_cumprod_prev": _compare(
            observed.sqrt_alphas_cumprod_prev,
            expected.sqrt_alphas_cumprod_prev,
            FLOAT64_RTOL,
            0.0,
        ),
    }
    failed = [name for name, result in checks.items() if not result["match"]]
    if failed:
        raise ScheduleMismatchError(
            f"Model schedule differs from the declared '{expected.description}' in: {failed}. "
            f"Details: {checks}"
        )
    return {
        **expected.summary().to_dict(),
        "observed_beta_last": float(observed.betas[-1]),
        "observed_alpha_bar_last": float(observed.alphas_cumprod[-1]),
        "checks": checks,
        "verified": True,
    }


def _compare(observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> dict:
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != expected.shape:
        return {"match": False, "reason": f"shape {observed.shape} != {expected.shape}"}
    relative = np.abs(observed - expected) / np.maximum(np.abs(expected), np.finfo(float).tiny)
    return {
        "match": bool(np.allclose(observed, expected, rtol=rtol, atol=atol)),
        "max_relative_error": float(relative.max()),
    }
