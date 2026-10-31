# ADDED: new file, not part of the upstream SR3 codebase.
"""Check that the schedule loaded inside the model equals the independently computed one."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import dataclass  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED

from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED

FLOAT32_RTOL = 1e-5  # ADDED
FLOAT32_ATOL = 1e-7  # ADDED
FLOAT64_RTOL = 1e-10  # ADDED


class ScheduleMismatchError(RuntimeError):  # ADDED
    """The model is not using the schedule the run declares."""  # ADDED


@dataclass(frozen=True)  # ADDED
class ScheduleBuffers:  # ADDED
    """Schedule buffers read back from a model.  # ADDED

    ``betas`` and ``alphas_cumprod`` are float32 tensors in SR3; ``sqrt_alphas_cumprod_prev`` is  # ADDED
    a float64 NumPy array that drives the continuous noise level fed to the UNet.  # ADDED
    """  # ADDED

    betas: np.ndarray  # ADDED
    alphas_cumprod: np.ndarray  # ADDED
    sqrt_alphas_cumprod_prev: np.ndarray  # ADDED


def verify_schedule(observed: ScheduleBuffers, expected: NoiseSchedule) -> dict[str, Any]:  # ADDED
    """Compare ``observed`` model buffers with ``expected`` and return the evidence record.  # ADDED

    Raises:  # ADDED
        ScheduleMismatchError: if any buffer differs beyond floating-point tolerance.  # ADDED
    """  # ADDED
    checks = {  # ADDED
        "betas": _compare(observed.betas, expected.betas, FLOAT32_RTOL, 0.0),  # ADDED
        "alphas_cumprod": _compare(  # ADDED
            observed.alphas_cumprod,  # ADDED
            expected.alphas_cumprod,  # ADDED
            FLOAT32_RTOL,  # ADDED
            FLOAT32_ATOL,  # ADDED
        ),  # ADDED
        "sqrt_alphas_cumprod_prev": _compare(  # ADDED
            observed.sqrt_alphas_cumprod_prev,  # ADDED
            expected.sqrt_alphas_cumprod_prev,  # ADDED
            FLOAT64_RTOL,  # ADDED
            0.0,  # ADDED
        ),  # ADDED
    }  # ADDED
    failed = [name for name, result in checks.items() if not result["match"]]  # ADDED
    if failed:  # ADDED
        raise ScheduleMismatchError(  # ADDED
            f"Model schedule differs from the declared '{expected.description}' in: {failed}. "  # ADDED
            f"Details: {checks}"  # ADDED
        )  # ADDED
    return {  # ADDED
        **expected.summary().to_dict(),  # ADDED
        "observed_beta_last": float(observed.betas[-1]),  # ADDED
        "observed_alpha_bar_last": float(observed.alphas_cumprod[-1]),  # ADDED
        "checks": checks,  # ADDED
        "verified": True,  # ADDED
    }  # ADDED


def _compare(observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> dict:  # ADDED
    observed = np.asarray(observed, dtype=np.float64)  # ADDED
    if observed.shape != expected.shape:  # ADDED
        return {"match": False, "reason": f"shape {observed.shape} != {expected.shape}"}  # ADDED
    relative = np.abs(observed - expected) / np.maximum(np.abs(expected), np.finfo(float).tiny)  # ADDED
    return {  # ADDED
        "match": bool(np.allclose(observed, expected, rtol=rtol, atol=atol)),  # ADDED
        "max_relative_error": float(relative.max()),  # ADDED
    }  # ADDED
