# ADDED: new file, not part of the upstream SR3 codebase.
"""Experiment configuration: TOML files mapped onto frozen dataclasses."""  # ADDED

from sr3_ablation.config.loader import build_config, load_config, save_resolved_config  # ADDED
from sr3_ablation.config.schema import (  # ADDED
    ConfigError,  # ADDED
    DataConfig,  # ADDED
    EvaluationConfig,  # ADDED
    ExperimentConfig,  # ADDED
    ModelConfig,  # ADDED
    PathsConfig,  # ADDED
    SamplingConfig,  # ADDED
    ScheduleConfig,  # ADDED
    TrainConfig,  # ADDED
    UNetConfig,  # ADDED
)  # ADDED

__all__ = [  # ADDED
    "ConfigError",  # ADDED
    "DataConfig",  # ADDED
    "EvaluationConfig",  # ADDED
    "ExperimentConfig",  # ADDED
    "ModelConfig",  # ADDED
    "PathsConfig",  # ADDED
    "SamplingConfig",  # ADDED
    "ScheduleConfig",  # ADDED
    "TrainConfig",  # ADDED
    "UNetConfig",  # ADDED
    "build_config",  # ADDED
    "load_config",  # ADDED
    "save_resolved_config",  # ADDED
]  # ADDED
