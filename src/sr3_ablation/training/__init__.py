# ADDED: new file, not part of the upstream SR3 codebase.
"""Fine-tuning loop and the callbacks that record its evidence."""

from sr3_ablation.training.callbacks import (
    CheckpointSaver,
    LossCsvLogger,
    RunRecordWriter,
    ScheduleEvidenceWriter,
    SnapshotWriter,
    ValidationCallback,
    ValidationMetricLogger,
    ValidationOutputs,
    ValidationSink,
)
from sr3_ablation.training.trainer import (
    EpochResult,
    FineTuner,
    NonFiniteLossError,
    TrainingCallback,
)

__all__ = [
    "CheckpointSaver",
    "EpochResult",
    "FineTuner",
    "LossCsvLogger",
    "NonFiniteLossError",
    "RunRecordWriter",
    "ScheduleEvidenceWriter",
    "SnapshotWriter",
    "TrainingCallback",
    "ValidationCallback",
    "ValidationMetricLogger",
    "ValidationOutputs",
    "ValidationSink",
]
