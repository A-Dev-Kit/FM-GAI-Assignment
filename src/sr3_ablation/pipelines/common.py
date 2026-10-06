# ADDED: new file, not part of the upstream SR3 codebase.
"""Shared set-up for every command: run folder, logging, seeding, device, datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sr3_ablation.config.loader import save_resolved_config
from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.loaders import PairedSrDataset
from sr3_ablation.data.preprocess import layout_for
from sr3_ablation.io.jsonio import read_json, write_json
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.runs import RunSpec
from sr3_ablation.schedule.noise_schedule import NoiseSchedule
from sr3_ablation.schedule.transforms import transform_for
from sr3_ablation.utils.device import resolve_device
from sr3_ablation.utils.env_info import collect_environment
from sr3_ablation.utils.log import configure_logging, get_logger
from sr3_ablation.utils.seeding import seed_everything

logger = get_logger("pipelines")


@dataclass(frozen=True)
class RunContext:
    config: ExperimentConfig
    run: RunSpec
    paths: RunPaths
    device: torch.device


def open_run(config: ExperimentConfig, run: RunSpec, command: str) -> RunContext:
    """Create the run folder, start logging and record environment and configuration."""
    paths = RunPaths.for_run(config.paths.output_root, run.name).create()
    configure_logging(paths.log_file)
    device = resolve_device(config.device)
    seed_everything(config.train.seed)
    write_json(collect_environment(device), paths.environment)
    save_resolved_config(config, paths.resolved_config)
    logger.info("%s | %s | exponent %.2f | device %s", command, run.label, run.exponent, device)
    return RunContext(config, run, paths, device)


def expected_schedule(config: ExperimentConfig, exponent: float) -> NoiseSchedule:
    return NoiseSchedule.from_config(config.schedule, transform_for(exponent))


def pretrained_weights(config: ExperimentConfig) -> Path:
    path = config.paths.checkpoint_dir / config.model.pretrained_file
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run: python -m sr3_ablation download-checkpoint"
        )
    return path


def weights_for(context: RunContext) -> Path:
    """Pretrained weights for Run 0, otherwise the run's final fine-tuned checkpoint."""
    if not context.run.fine_tuned:
        return pretrained_weights(context.config)
    final = context.paths.final_checkpoint()
    if not final.is_file():
        command = f"python scripts/finetune.py --run {context.run.key}"
        raise FileNotFoundError(f"{final} not found; fine-tune first: {command}")
    check_trained_exponent(context)
    return final


def check_trained_exponent(context: RunContext) -> None:
    """Refuse to sample with an exponent different from the one the run was trained with."""
    if not context.paths.run_record.is_file():
        return
    trained = float(read_json(context.paths.run_record)["exponent"])
    if abs(trained - context.run.exponent) > 1e-12:
        raise ValueError(
            f"{context.run.label} was fine-tuned with x = {trained} but NOISE_EXPONENT_X is now "
            f"{context.run.exponent}; restore the constant or fine-tune again."
        )


def load_split(
    config: ExperimentConfig, split: str, *, augment: bool = False, limit: int = 0
) -> PairedSrDataset:
    layout = layout_for(
        config.paths.data_root, split, config.data.low_resolution, config.data.high_resolution
    )
    return PairedSrDataset(layout, augment=augment, limit=limit)


def collect_batches(dataset: PairedSrDataset, batch_size: int) -> list[dict[str, Any]]:
    """Materialise a small dataset as a list of batches (used for fixed image sets)."""
    return list(DataLoader(dataset, batch_size=batch_size, shuffle=False))
