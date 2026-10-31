# ADDED: new file, not part of the upstream SR3 codebase.
"""Fine-tuning loop and the callbacks that record its evidence."""  # ADDED

from sr3_ablation.training.callbacks import (  # ADDED
    CheckpointSaver,  # ADDED
    LossCsvLogger,  # ADDED
    RunRecordWriter,  # ADDED
    ScheduleEvidenceWriter,  # ADDED
    SnapshotWriter,  # ADDED
    ValidationCallback,  # ADDED
    ValidationMetricLogger,  # ADDED
    ValidationOutputs,  # ADDED
    ValidationSink,  # ADDED
)  # ADDED
from sr3_ablation.training.trainer import (  # ADDED
    EpochResult,  # ADDED
    FineTuner,  # ADDED
    NonFiniteLossError,  # ADDED
    TrainingCallback,  # ADDED
)  # ADDED

__all__ = [  # ADDED
    "CheckpointSaver",  # ADDED
    "EpochResult",  # ADDED
    "FineTuner",  # ADDED
    "LossCsvLogger",  # ADDED
    "NonFiniteLossError",  # ADDED
    "RunRecordWriter",  # ADDED
    "ScheduleEvidenceWriter",  # ADDED
    "SnapshotWriter",  # ADDED
    "TrainingCallback",  # ADDED
    "ValidationCallback",  # ADDED
    "ValidationMetricLogger",  # ADDED
    "ValidationOutputs",  # ADDED
    "ValidationSink",  # ADDED
]  # ADDED
