# ADDED: new file, not part of the upstream SR3 codebase.
"""Command smoke-test: a short end-to-end run that also measures speed and memory.

Runs schedule verification, Run 0 on a few test images, a capped Run B fine-tune, Run B
evaluation and trajectories, then extrapolates the cost of the full experiment.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

from sr3_ablation.config.loader import load_config
from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.pipelines.common import load_split
from sr3_ablation.pipelines.evaluate import evaluate_run, record_trajectories
from sr3_ablation.pipelines.finetune import finetune, verify_run_schedule
from sr3_ablation.runs import RunSpec, run_spec
from sr3_ablation.training.trainer import EpochResult
from sr3_ablation.utils.device import peak_memory_mib, reset_peak_memory, resolve_device, timed
from sr3_ablation.utils.log import get_logger

FALLBACK_BATCH = (4, 2)

logger = get_logger("pipelines.smoke")


def smoke_test(config: ExperimentConfig, exponent_x: float) -> dict[str, Any]:
    device = resolve_device(config.device)
    run_a, run_b = run_spec("A", exponent_x), run_spec("B", exponent_x)
    timings: dict[str, Any] = {"device": str(device), "exponent": exponent_x}

    for run in (run_a, run_b):
        verify_run_schedule(config, run)

    with timed(device) as run0_time:
        evaluate_run(config, run_spec("0", exponent_x))
    test_images = len(load_split(config, "test", limit=config.evaluation.test_limit))
    sampling_batches = math.ceil(test_images / config.sampling.batch_size)
    timings["run0_test_images"] = test_images
    timings["run0_eval_seconds"] = run0_time["seconds"]
    timings["sampling_seconds_per_batch"] = run0_time["seconds"] / sampling_batches

    reset_peak_memory(device)
    config, results = _finetune_with_fallback(config, run_b)
    epoch = results[-1]
    timings["train_batch_size"] = config.train.batch_size
    timings["train_grad_accum_steps"] = config.train.grad_accum_steps
    timings["train_iterations"] = epoch.iterations
    timings["train_seconds_per_iteration"] = epoch.seconds / epoch.iterations
    timings["train_peak_memory_mib"] = peak_memory_mib(device)

    with timed(device) as run_b_time:
        evaluate_run(config, run_b)
    timings["runB_eval_seconds"] = run_b_time["seconds"]
    with timed(device) as trajectory_time:
        record_trajectories(config, run_b)
    timings["trajectory_seconds"] = trajectory_time["seconds"]

    timings["estimates_for_full_run"] = _estimate_full_run(timings)
    write_json(timings, config.paths.output_root / "timings.json")
    logger.info("Smoke test finished: %s", timings)
    return timings


def _finetune_with_fallback(
    config: ExperimentConfig, run: RunSpec
) -> tuple[ExperimentConfig, list[EpochResult]]:
    """Fine-tune; on CUDA out-of-memory retry with batch 4 and 2 accumulation steps."""
    try:
        return config, finetune(config, run)
    except torch.cuda.OutOfMemoryError:
        batch, accum = FALLBACK_BATCH
        logger.warning(
            "Out of memory at batch %d; retrying with %d x %d.",
            config.train.batch_size,
            batch,
            accum,
        )
        torch.cuda.empty_cache()
        smaller = dataclasses.replace(
            config,
            train=dataclasses.replace(config.train, batch_size=batch, grad_accum_steps=accum),
        )
        return smaller, finetune(smaller, run)


def _estimate_full_run(timings: dict[str, Any]) -> dict[str, float]:
    """Hours for the full experiment of ``configs/base.toml`` on this machine.

    Assumes the measured per-iteration and per-batch costs, i.e. the batch size that fitted
    in the smoke test and the same sampling batch size.
    """
    full = load_config()
    train_images = len(load_split(full, "train", limit=full.data.train_limit))
    test_images = len(load_split(full, "test", limit=full.evaluation.test_limit))
    iterations_per_epoch = math.ceil(train_images / timings["train_batch_size"])
    per_batch = timings["sampling_seconds_per_batch"]
    sampling_batches = math.ceil(full.sampling.validation_images / full.sampling.batch_size)
    train_hours = (
        full.train.epochs * iterations_per_epoch * timings["train_seconds_per_iteration"] / 3600
    )
    validation_hours = len(full.train.checkpoint_epochs) * sampling_batches * per_batch / 3600
    test_hours = math.ceil(test_images / full.sampling.batch_size) * per_batch / 3600
    return {
        "train_images": train_images,
        "iterations_per_epoch": iterations_per_epoch,
        "finetune_hours_per_run": round(train_hours, 2),
        "validation_sampling_hours_per_run": round(validation_hours, 2),
        "test_evaluation_hours_per_run": round(test_hours, 2),
        "total_hours_runs_0_A_B": round(2 * (train_hours + validation_hours) + 3 * test_hours, 2),
    }
