# ADDED: new file, not part of the upstream SR3 codebase.
"""Shared set-up for every command: run folder, logging, seeding, device, datasets."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

import torch  # ADDED
from torch.utils.data import DataLoader  # ADDED

from sr3_ablation.config.loader import save_resolved_config  # ADDED
from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.data.loaders import PairedSrDataset  # ADDED
from sr3_ablation.data.preprocess import layout_for  # ADDED
from sr3_ablation.io.jsonio import read_json, write_json  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.runs import PowerTarget, RunSpec  # ADDED
from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED
from sr3_ablation.schedule.transforms import transform_for  # ADDED
from sr3_ablation.utils.device import resolve_device  # ADDED
from sr3_ablation.utils.env_info import collect_environment  # ADDED
from sr3_ablation.utils.log import configure_logging, get_logger  # ADDED
from sr3_ablation.utils.seeding import seed_everything  # ADDED

logger = get_logger("pipelines")  # ADDED


@dataclass(frozen=True)  # ADDED
class RunContext:  # ADDED
    """Everything a command needs to know about the run it works on."""  # ADDED

    config: ExperimentConfig  # ADDED
    run: RunSpec  # ADDED
    paths: RunPaths  # ADDED
    device: torch.device  # ADDED


def open_run(config: ExperimentConfig, run: RunSpec, command: str) -> RunContext:  # ADDED
    """Create the run folder, start logging and record environment and configuration."""  # ADDED
    paths = RunPaths.for_run(config.paths.output_root, run.name).create()  # ADDED
    configure_logging(paths.log_file)  # ADDED
    device = resolve_device(config.device)  # ADDED
    seed_everything(config.train.seed)  # ADDED
    write_json(collect_environment(device), paths.environment)  # ADDED
    save_resolved_config(config, paths.resolved_config)  # ADDED
    logger.info("%s | %s | exponent %.2f | device %s", command, run.label, run.exponent, device)  # ADDED
    return RunContext(config, run, paths, device)  # ADDED


def expected_schedule(config: ExperimentConfig, run: RunSpec) -> NoiseSchedule:  # ADDED
    """The schedule a run should use, computed independently of SR3."""  # ADDED
    return NoiseSchedule.from_config(  # ADDED
        config.schedule,  # ADDED
        transform_for(run.exponent, power_on=run.power_target),  # ADDED
    )  # ADDED


def expected_schedule_for_exponent(  # ADDED
    config: ExperimentConfig,  # ADDED
    exponent: float,  # ADDED
    *,  # ADDED
    power_on: PowerTarget = "beta",  # ADDED
) -> NoiseSchedule:  # ADDED
    """Backward-compatible helper when only the exponent is known."""  # ADDED
    return NoiseSchedule.from_config(config.schedule, transform_for(exponent, power_on=power_on))  # ADDED


def pretrained_weights(config: ExperimentConfig) -> Path:  # ADDED
    """Path of the pretrained SR3 checkpoint; fails with the download command if missing."""  # ADDED
    path = config.paths.checkpoint_dir / config.model.pretrained_file  # ADDED
    if not path.is_file():  # ADDED
        raise FileNotFoundError(  # ADDED
            f"{path} not found; run: python -m sr3_ablation download-checkpoint"  # ADDED
        )  # ADDED
    return path  # ADDED


def weights_for(context: RunContext) -> Path:  # ADDED
    """Pretrained weights for Run 0, otherwise the run's final fine-tuned checkpoint."""  # ADDED
    if not context.run.fine_tuned:  # ADDED
        return pretrained_weights(context.config)  # ADDED
    final = context.paths.final_checkpoint()  # ADDED
    if not final.is_file():  # ADDED
        command = f"python scripts/finetune.py --run {context.run.key}"  # ADDED
        raise FileNotFoundError(f"{final} not found; fine-tune first: {command}")  # ADDED
    check_trained_exponent(context)  # ADDED
    return final  # ADDED


def check_trained_exponent(context: RunContext) -> None:  # ADDED
    """Refuse to sample with an exponent different from the one the run was trained with."""  # ADDED
    if not context.paths.run_record.is_file():  # ADDED
        return  # ADDED
    trained = read_json(context.paths.run_record)  # ADDED
    trained_exponent = float(trained["exponent"])  # ADDED
    if abs(trained_exponent - context.run.exponent) > 1e-12:  # ADDED
        raise ValueError(  # ADDED
            f"{context.run.label} was fine-tuned with x = {trained_exponent} but the run "  # ADDED
            f"requests x = {context.run.exponent}."  # ADDED
        )  # ADDED
    if trained.get("power_target", "beta") != context.run.power_target:  # ADDED
        raise ValueError(  # ADDED
            f"{context.run.label} was trained with power_target={trained.get('power_target')} "  # ADDED
            f"but this run uses {context.run.power_target}."  # ADDED
        )  # ADDED


def load_split(  # ADDED
    config: ExperimentConfig,  # ADDED
    split: str,  # ADDED
    *,  # ADDED
    augment: bool = False,  # ADDED
    limit: int = 0,  # ADDED
) -> PairedSrDataset:  # ADDED
    """Paired LR/HR dataset of one processed split."""  # ADDED
    layout = layout_for(  # ADDED
        config.paths.data_root,  # ADDED
        split,  # ADDED
        config.data.low_resolution,  # ADDED
        config.data.high_resolution,  # ADDED
    )  # ADDED
    return PairedSrDataset(layout, augment=augment, limit=limit)  # ADDED


def collect_batches(dataset: PairedSrDataset, batch_size: int) -> list[dict[str, Any]]:  # ADDED
    """Materialise a small dataset as a list of batches (used for fixed image sets)."""  # ADDED
    return list(DataLoader(dataset, batch_size=batch_size, shuffle=False))  # ADDED
