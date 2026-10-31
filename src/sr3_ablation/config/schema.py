# ADDED: new file, not part of the upstream SR3 codebase.
"""Typed, immutable experiment configuration."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import asdict, dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED


class ConfigError(ValueError):  # ADDED
    """Raised when a configuration file is incomplete or inconsistent."""  # ADDED


@dataclass(frozen=True)  # ADDED
class PathsConfig:  # ADDED
    """Where the data, run outputs and checkpoints live (absolute after loading)."""  # ADDED

    data_root: Path  # ADDED
    output_root: Path  # ADDED
    checkpoint_dir: Path  # ADDED


@dataclass(frozen=True)  # ADDED
class DataConfig:  # ADDED
    """Kvasir-SEG source, seeded split sizes and the LR/HR resolutions."""  # ADDED

    source_url: str  # ADDED
    split_seed: int  # ADDED
    val_size: int  # ADDED
    test_size: int  # ADDED
    low_resolution: int  # ADDED
    high_resolution: int  # ADDED
    train_limit: int  # ADDED
    num_workers: int  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if self.low_resolution >= self.high_resolution:  # ADDED
            raise ConfigError("data.low_resolution must be smaller than data.high_resolution.")  # ADDED
        if self.val_size <= 0 or self.test_size <= 0 or self.train_limit < 0:  # ADDED
            raise ConfigError("data.val_size and data.test_size must be positive.")  # ADDED


@dataclass(frozen=True)  # ADDED
class ScheduleConfig:  # ADDED
    """SR3's base noise schedule, before any power transform."""  # ADDED

    schedule: str  # ADDED
    n_timestep: int  # ADDED
    linear_start: float  # ADDED
    linear_end: float  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if not 0.0 < self.linear_start < self.linear_end < 1.0:  # ADDED
            raise ConfigError("schedule requires 0 < linear_start < linear_end < 1.")  # ADDED
        if self.n_timestep < 2:  # ADDED
            raise ConfigError("schedule.n_timestep must be at least 2.")  # ADDED

    def to_upstream(  # ADDED
        self,  # ADDED
        exponent: float = 1.0,  # ADDED
        *,  # ADDED
        alpha_bar_exponent: float = 1.0,  # ADDED
    ) -> dict[str, Any]:  # ADDED
        """Dictionary in the format expected by SR3 ``set_new_noise_schedule``."""  # ADDED
        return {  # ADDED
            "schedule": self.schedule,  # ADDED
            "n_timestep": self.n_timestep,  # ADDED
            "linear_start": self.linear_start,  # ADDED
            "linear_end": self.linear_end,  # ADDED
            "exponent": exponent,  # ADDED
            "alpha_bar_exponent": alpha_bar_exponent,  # ADDED
        }  # ADDED


@dataclass(frozen=True)  # ADDED
class UNetConfig:  # ADDED
    """Constructor arguments of SR3's UNet."""  # ADDED

    in_channel: int  # ADDED
    out_channel: int  # ADDED
    inner_channel: int  # ADDED
    norm_groups: int  # ADDED
    channel_multiplier: tuple[int, ...]  # ADDED
    attn_res: tuple[int, ...]  # ADDED
    res_blocks: int  # ADDED
    dropout: float  # ADDED


@dataclass(frozen=True)  # ADDED
class ModelConfig:  # ADDED
    """Pretrained checkpoint location and the SR3 model settings."""  # ADDED

    pretrained_folder_url: str  # ADDED
    pretrained_file: str  # ADDED
    image_size: int  # ADDED
    channels: int  # ADDED
    unet: UNetConfig  # ADDED


@dataclass(frozen=True)  # ADDED
class TrainConfig:  # ADDED
    """Fine-tuning hyperparameters, shared by Runs A and B."""  # ADDED

    seed: int  # ADDED
    batch_size: int  # ADDED
    grad_accum_steps: int  # ADDED
    learning_rate: float  # ADDED
    epochs: int  # ADDED
    checkpoint_epochs: tuple[int, ...]  # ADDED
    max_iterations: int  # ADDED
    log_every: int  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if self.batch_size < 1 or self.grad_accum_steps < 1 or self.epochs < 1:  # ADDED
            raise ConfigError("train.batch_size, grad_accum_steps and epochs must be >= 1.")  # ADDED
        if not self.checkpoint_epochs:  # ADDED
            raise ConfigError("train.checkpoint_epochs must list at least one epoch.")  # ADDED
        if any(not 1 <= epoch <= self.epochs for epoch in self.checkpoint_epochs):  # ADDED
            raise ConfigError("train.checkpoint_epochs must lie in [1, train.epochs].")  # ADDED
        if self.max_iterations < 0:  # ADDED
            raise ConfigError("train.max_iterations must be >= 0 (0 means no limit).")  # ADDED

    @property  # ADDED
    def effective_batch_size(self) -> int:  # ADDED
        return self.batch_size * self.grad_accum_steps  # ADDED


@dataclass(frozen=True)  # ADDED
class SamplingConfig:  # ADDED
    """Seeds, batch size and the fixed image sets for snapshots and trajectories."""  # ADDED

    seed: int  # ADDED
    batch_size: int  # ADDED
    snapshot_samples: int  # ADDED
    validation_images: int  # ADDED
    trajectory_samples: int  # ADDED
    trajectory_timesteps: tuple[int, ...]  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if self.snapshot_samples > self.validation_images:  # ADDED
            raise ConfigError("sampling.snapshot_samples cannot exceed validation_images.")  # ADDED


@dataclass(frozen=True)  # ADDED
class EvaluationConfig:  # ADDED
    """Test metrics and an optional cap on the number of test images (0 = all)."""  # ADDED

    metrics: tuple[str, ...]  # ADDED
    test_limit: int  # ADDED


@dataclass(frozen=True)  # ADDED
class ExperimentConfig:  # ADDED
    """The complete, validated configuration of one experiment."""  # ADDED

    name: str  # ADDED
    device: str  # ADDED
    paths: PathsConfig  # ADDED
    data: DataConfig  # ADDED
    schedule: ScheduleConfig  # ADDED
    model: ModelConfig  # ADDED
    train: TrainConfig  # ADDED
    sampling: SamplingConfig  # ADDED
    evaluation: EvaluationConfig  # ADDED

    def __post_init__(self) -> None:  # ADDED
        horizon = self.schedule.n_timestep  # ADDED
        if any(not 0 <= t <= horizon for t in self.sampling.trajectory_timesteps):  # ADDED
            raise ConfigError(f"sampling.trajectory_timesteps must lie in [0, {horizon}].")  # ADDED
        if self.model.image_size != self.data.high_resolution:  # ADDED
            raise ConfigError("model.image_size must equal data.high_resolution.")  # ADDED

    def to_dict(self) -> dict[str, Any]:  # ADDED
        """JSON-serialisable view of the configuration (paths become strings)."""  # ADDED
        return _stringify_paths(asdict(self))  # ADDED


def _stringify_paths(value: Any) -> Any:  # ADDED
    if isinstance(value, dict):  # ADDED
        return {key: _stringify_paths(item) for key, item in value.items()}  # ADDED
    if isinstance(value, list | tuple):  # ADDED
        return [_stringify_paths(item) for item in value]  # ADDED
    if isinstance(value, Path):  # ADDED
        return value.as_posix()  # ADDED
    return value  # ADDED
