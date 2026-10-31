# ADDED: new file, not part of the upstream SR3 codebase.
"""Model interfaces, the SR3 adapter and checkpoint handling."""  # ADDED

from sr3_ablation.model.checkpoints import (  # ADDED
    download_pretrained,  # ADDED
    read_denoiser_state,  # ADDED
    save_denoiser_state,  # ADDED
)  # ADDED
from sr3_ablation.model.factory import build_backend  # ADDED
from sr3_ablation.model.protocols import (  # ADDED
    Denoiser,  # ADDED
    DiffusionBackend,  # ADDED
    ScheduleHolder,  # ADDED
    TrainableModel,  # ADDED
    WeightStore,  # ADDED
)  # ADDED
from sr3_ablation.model.sr3_backend import Sr3Backend  # ADDED

__all__ = [  # ADDED
    "Denoiser",  # ADDED
    "DiffusionBackend",  # ADDED
    "ScheduleHolder",  # ADDED
    "Sr3Backend",  # ADDED
    "TrainableModel",  # ADDED
    "WeightStore",  # ADDED
    "build_backend",  # ADDED
    "download_pretrained",  # ADDED
    "read_denoiser_state",  # ADDED
    "save_denoiser_state",  # ADDED
]  # ADDED
