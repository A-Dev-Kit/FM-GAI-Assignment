# ADDED: new file, not part of the upstream SR3 codebase.
"""Load layered TOML files into an :class:`ExperimentConfig`."""  # ADDED

from __future__ import annotations  # ADDED

import dataclasses  # ADDED
import tomllib  # ADDED
from collections.abc import Mapping, Sequence  # ADDED
from pathlib import Path  # ADDED
from typing import Any, TypeVar  # ADDED

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
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.paths import DEFAULT_CONFIG, LOCAL_CONFIG, PROJECT_ROOT  # ADDED

SectionT = TypeVar("SectionT")  # ADDED


def load_config(  # ADDED
    paths: Sequence[Path] | None = None,  # ADDED
    project_root: Path = PROJECT_ROOT,  # ADDED
    local_config: Path | None = LOCAL_CONFIG,  # ADDED
) -> ExperimentConfig:  # ADDED
    """Merge the TOML files in ``paths`` (later files win) and build the configuration.  # ADDED

    A machine-specific ``configs/local.toml`` (git-ignored, e.g. to keep data outside a  # ADDED
    cloud-synced folder) is merged last when it exists. Relative paths inside ``[paths]`` are  # ADDED
    resolved against ``project_root``.  # ADDED
    """  # ADDED
    files = [Path(path) for path in paths or [DEFAULT_CONFIG]]  # ADDED
    if local_config is not None and local_config.is_file():  # ADDED
        files.append(local_config)  # ADDED
    merged: dict[str, Any] = {}  # ADDED
    for path in files:  # ADDED
        merged = deep_merge(merged, read_toml(path))  # ADDED
    return build_config(merged, project_root)  # ADDED


def read_toml(path: Path) -> dict[str, Any]:  # ADDED
    """Parse one TOML file; a missing file is a :class:`ConfigError`."""  # ADDED
    if not path.is_file():  # ADDED
        raise ConfigError(f"Configuration file not found: {path}")  # ADDED
    with path.open("rb") as handle:  # ADDED
        return tomllib.load(handle)  # ADDED


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:  # ADDED
    """Recursively merge ``override`` into a copy of ``base``."""  # ADDED
    result = dict(base)  # ADDED
    for key, value in override.items():  # ADDED
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):  # ADDED
            result[key] = deep_merge(result[key], value)  # ADDED
        else:  # ADDED
            result[key] = value  # ADDED
    return result  # ADDED


def build_config(raw: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> ExperimentConfig:  # ADDED
    """Validate a merged raw mapping; relative paths resolve against ``project_root``."""  # ADDED
    experiment = _require(raw, "experiment")  # ADDED
    model_raw = dict(_require(raw, "model"))  # ADDED
    unet = _section(UNetConfig, _require(model_raw, "unet"), "model.unet")  # ADDED
    model_raw.pop("unet")  # ADDED
    paths_raw = {  # ADDED
        key: _resolve(value, project_root)  # ADDED
        for key, value in _require(raw, "paths").items()  # ADDED
    }  # ADDED
    return ExperimentConfig(  # ADDED
        name=str(_require(experiment, "name")),  # ADDED
        device=str(experiment.get("device", "auto")),  # ADDED
        paths=_section(PathsConfig, paths_raw, "paths"),  # ADDED
        data=_section(DataConfig, _require(raw, "data"), "data"),  # ADDED
        schedule=_section(ScheduleConfig, _require(raw, "schedule"), "schedule"),  # ADDED
        model=_section(ModelConfig, {**model_raw, "unet": unet}, "model"),  # ADDED
        train=_section(TrainConfig, _require(raw, "train"), "train"),  # ADDED
        sampling=_section(SamplingConfig, _require(raw, "sampling"), "sampling"),  # ADDED
        evaluation=_section(EvaluationConfig, _require(raw, "evaluation"), "evaluation"),  # ADDED
    )  # ADDED


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:  # ADDED
    """Write the fully merged configuration next to a run's outputs."""  # ADDED
    write_json(config.to_dict(), path)  # ADDED


def _section(cls: type[SectionT], values: Mapping[str, Any], name: str) -> SectionT:  # ADDED
    expected = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]  # ADDED
    unknown = set(values) - expected  # ADDED
    missing = expected - set(values)  # ADDED
    if unknown:  # ADDED
        raise ConfigError(f"[{name}] has unknown keys: {sorted(unknown)}")  # ADDED
    if missing:  # ADDED
        raise ConfigError(f"[{name}] is missing keys: {sorted(missing)}")  # ADDED
    converted = {key: tuple(val) if isinstance(val, list) else val for key, val in values.items()}  # ADDED
    return cls(**converted)  # ADDED


def _require(mapping: Mapping[str, Any], key: str) -> Any:  # ADDED
    if key not in mapping:  # ADDED
        raise ConfigError(f"Missing configuration section or key: {key!r}")  # ADDED
    return mapping[key]  # ADDED


def _resolve(value: str, project_root: Path) -> Path:  # ADDED
    path = Path(value)  # ADDED
    return path if path.is_absolute() else (project_root / path).resolve()  # ADDED
