# ADDED: new file, not part of the upstream SR3 codebase.
"""Seed every random number generator used by training and sampling."""  # ADDED

from __future__ import annotations  # ADDED

import os  # ADDED
import random  # ADDED

import numpy as np  # ADDED
import torch  # ADDED


def seed_everything(seed: int) -> None:  # ADDED
    """Seed Python, NumPy (used by SR3 to draw timesteps) and PyTorch (CPU and CUDA)."""  # ADDED
    os.environ["PYTHONHASHSEED"] = str(seed)  # ADDED
    random.seed(seed)  # ADDED
    np.random.seed(seed)  # ADDED
    torch.manual_seed(seed)  # ADDED
    torch.cuda.manual_seed_all(seed)  # ADDED
