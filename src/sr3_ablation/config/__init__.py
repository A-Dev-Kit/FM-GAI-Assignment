# ADDED: new file, not part of the upstream SR3 codebase.
"""Experiment configuration: TOML files mapped onto frozen dataclasses."""

from sr3_ablation.config.loader import build_config, load_config, save_resolved_config
from sr3_ablation.config.schema import (
    ConfigError,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    SamplingConfig,
    ScheduleConfig,
    TrainConfig,
    UNetConfig,
)

__all__ = [
    "ConfigError",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathsConfig",
    "SamplingConfig",
    "ScheduleConfig",
    "TrainConfig",
    "UNetConfig",
    "build_config",
    "load_config",
    "save_resolved_config",
]
