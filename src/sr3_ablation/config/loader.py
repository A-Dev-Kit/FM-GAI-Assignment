# ADDED: new file, not part of the upstream SR3 codebase.
"""Load layered TOML files into an :class:`ExperimentConfig`."""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

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
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.paths import DEFAULT_CONFIG, LOCAL_CONFIG, PROJECT_ROOT

SectionT = TypeVar("SectionT")


def load_config(
    paths: Sequence[Path] | None = None,
    project_root: Path = PROJECT_ROOT,
    local_config: Path | None = LOCAL_CONFIG,
) -> ExperimentConfig:
    """Merge the TOML files in ``paths`` (later files win) and build the configuration.

    A machine-specific ``configs/local.toml`` (git-ignored, e.g. to keep data outside a
    cloud-synced folder) is merged last when it exists. Relative paths inside ``[paths]`` are
    resolved against ``project_root``.
    """
    files = [Path(path) for path in paths or [DEFAULT_CONFIG]]
    if local_config is not None and local_config.is_file():
        files.append(local_config)
    merged: dict[str, Any] = {}
    for path in files:
        merged = deep_merge(merged, read_toml(path))
    return build_config(merged, project_root)


def read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_config(raw: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> ExperimentConfig:
    experiment = _require(raw, "experiment")
    model_raw = dict(_require(raw, "model"))
    unet = _section(UNetConfig, _require(model_raw, "unet"), "model.unet")
    model_raw.pop("unet")
    paths_raw = {
        key: _resolve(value, project_root) for key, value in _require(raw, "paths").items()
    }
    return ExperimentConfig(
        name=str(_require(experiment, "name")),
        device=str(experiment.get("device", "auto")),
        paths=_section(PathsConfig, paths_raw, "paths"),
        data=_section(DataConfig, _require(raw, "data"), "data"),
        schedule=_section(ScheduleConfig, _require(raw, "schedule"), "schedule"),
        model=_section(ModelConfig, {**model_raw, "unet": unet}, "model"),
        train=_section(TrainConfig, _require(raw, "train"), "train"),
        sampling=_section(SamplingConfig, _require(raw, "sampling"), "sampling"),
        evaluation=_section(EvaluationConfig, _require(raw, "evaluation"), "evaluation"),
    )


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:
    """Write the fully merged configuration next to a run's outputs."""
    write_json(config.to_dict(), path)


def _section(cls: type[SectionT], values: Mapping[str, Any], name: str) -> SectionT:
    expected = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"[{name}] is missing keys: {sorted(missing)}")
    converted = {key: tuple(val) if isinstance(val, list) else val for key, val in values.items()}
    return cls(**converted)


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing configuration section or key: {key!r}")
    return mapping[key]


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()
