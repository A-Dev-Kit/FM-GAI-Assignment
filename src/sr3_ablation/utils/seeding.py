# ADDED: new file, not part of the upstream SR3 codebase.
"""Seed every random number generator used by training and sampling."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (used by SR3 to draw timesteps) and PyTorch (CPU and CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
