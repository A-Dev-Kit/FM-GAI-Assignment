# ADDED: new file, not part of the upstream SR3 codebase.
"""Typed, immutable experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is incomplete or inconsistent."""


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    output_root: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class DataConfig:
    source_url: str
    split_seed: int
    val_size: int
    test_size: int
    low_resolution: int
    high_resolution: int
    train_limit: int
    num_workers: int

    def __post_init__(self) -> None:
        if self.low_resolution >= self.high_resolution:
            raise ConfigError("data.low_resolution must be smaller than data.high_resolution.")
        if self.val_size <= 0 or self.test_size <= 0 or self.train_limit < 0:
            raise ConfigError("data.val_size and data.test_size must be positive.")


@dataclass(frozen=True)
class ScheduleConfig:
    schedule: str
    n_timestep: int
    linear_start: float
    linear_end: float

    def __post_init__(self) -> None:
        if not 0.0 < self.linear_start < self.linear_end < 1.0:
            raise ConfigError("schedule requires 0 < linear_start < linear_end < 1.")
        if self.n_timestep < 2:
            raise ConfigError("schedule.n_timestep must be at least 2.")

    def to_upstream(self, exponent: float = 1.0) -> dict[str, Any]:
        """Dictionary in the format expected by SR3 ``set_new_noise_schedule``."""
        return {
            "schedule": self.schedule,
            "n_timestep": self.n_timestep,
            "linear_start": self.linear_start,
            "linear_end": self.linear_end,
            "exponent": exponent,
        }


@dataclass(frozen=True)
class UNetConfig:
    in_channel: int
    out_channel: int
    inner_channel: int
    norm_groups: int
    channel_multiplier: tuple[int, ...]
    attn_res: tuple[int, ...]
    res_blocks: int
    dropout: float


@dataclass(frozen=True)
class ModelConfig:
    pretrained_folder_url: str
    pretrained_file: str
    image_size: int
    channels: int
    unet: UNetConfig


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    epochs: int
    checkpoint_epochs: tuple[int, ...]
    max_iterations: int
    log_every: int

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.grad_accum_steps < 1 or self.epochs < 1:
            raise ConfigError("train.batch_size, grad_accum_steps and epochs must be >= 1.")
        if not self.checkpoint_epochs:
            raise ConfigError("train.checkpoint_epochs must list at least one epoch.")
        if any(not 1 <= epoch <= self.epochs for epoch in self.checkpoint_epochs):
            raise ConfigError("train.checkpoint_epochs must lie in [1, train.epochs].")
        if self.max_iterations < 0:
            raise ConfigError("train.max_iterations must be >= 0 (0 means no limit).")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    batch_size: int
    snapshot_samples: int
    validation_images: int
    trajectory_samples: int
    trajectory_timesteps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.snapshot_samples > self.validation_images:
            raise ConfigError("sampling.snapshot_samples cannot exceed validation_images.")


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...]
    test_limit: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    device: str
    paths: PathsConfig
    data: DataConfig
    schedule: ScheduleConfig
    model: ModelConfig
    train: TrainConfig
    sampling: SamplingConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        horizon = self.schedule.n_timestep
        if any(not 0 <= t <= horizon for t in self.sampling.trajectory_timesteps):
            raise ConfigError(f"sampling.trajectory_timesteps must lie in [0, {horizon}].")
        if self.model.image_size != self.data.high_resolution:
            raise ConfigError("model.image_size must equal data.high_resolution.")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration (paths become strings)."""
        return _stringify_paths(asdict(self))


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
