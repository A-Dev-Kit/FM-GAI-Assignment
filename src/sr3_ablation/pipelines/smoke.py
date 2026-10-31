# ADDED: new file, not part of the upstream SR3 codebase.
"""Command smoke-test: a short end-to-end run that also measures speed and memory.  # ADDED

Runs schedule verification, Run 0 on a few test images, a capped Run B fine-tune, Run B  # ADDED
evaluation and trajectories, then extrapolates the cost of the full experiment.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

import dataclasses  # ADDED
import math  # ADDED
from typing import Any  # ADDED

import torch  # ADDED

from sr3_ablation.config.loader import load_config  # ADDED
from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.pipelines.common import load_split  # ADDED
from sr3_ablation.pipelines.evaluate import evaluate_run, record_trajectories  # ADDED
from sr3_ablation.pipelines.finetune import finetune, verify_run_schedule  # ADDED
from sr3_ablation.runs import RunSpec, run_spec  # ADDED
from sr3_ablation.training.trainer import EpochResult  # ADDED
from sr3_ablation.utils.device import peak_memory_mib, reset_peak_memory, resolve_device, timed  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

FALLBACK_BATCH = (4, 2)  # ADDED

logger = get_logger("pipelines.smoke")  # ADDED


def smoke_test(config: ExperimentConfig, exponent_x: float) -> dict[str, Any]:  # ADDED
    """Command ``smoke-test``: run every stage briefly and estimate full-run timings."""  # ADDED
    device = resolve_device(config.device)  # ADDED
    run_a, run_b = run_spec("A", exponent_x), run_spec("B", exponent_x)  # ADDED
    timings: dict[str, Any] = {"device": str(device), "exponent": exponent_x}  # ADDED

    for run in (run_a, run_b):  # ADDED
        verify_run_schedule(config, run)  # ADDED

    with timed(device) as run0_time:  # ADDED
        evaluate_run(config, run_spec("0", exponent_x))  # ADDED
    test_images = len(load_split(config, "test", limit=config.evaluation.test_limit))  # ADDED
    sampling_batches = math.ceil(test_images / config.sampling.batch_size)  # ADDED
    timings["run0_test_images"] = test_images  # ADDED
    timings["run0_eval_seconds"] = run0_time["seconds"]  # ADDED
    timings["sampling_seconds_per_batch"] = run0_time["seconds"] / sampling_batches  # ADDED

    reset_peak_memory(device)  # ADDED
    config, results = _finetune_with_fallback(config, run_b)  # ADDED
    epoch = results[-1]  # ADDED
    timings["train_batch_size"] = config.train.batch_size  # ADDED
    timings["train_grad_accum_steps"] = config.train.grad_accum_steps  # ADDED
    timings["train_iterations"] = epoch.iterations  # ADDED
    timings["train_seconds_per_iteration"] = epoch.seconds / epoch.iterations  # ADDED
    timings["train_peak_memory_mib"] = peak_memory_mib(device)  # ADDED

    with timed(device) as run_b_time:  # ADDED
        evaluate_run(config, run_b)  # ADDED
    timings["runB_eval_seconds"] = run_b_time["seconds"]  # ADDED
    with timed(device) as trajectory_time:  # ADDED
        record_trajectories(config, run_b)  # ADDED
    timings["trajectory_seconds"] = trajectory_time["seconds"]  # ADDED

    timings["estimates_for_full_run"] = _estimate_full_run(timings)  # ADDED
    write_json(timings, config.paths.output_root / "timings.json")  # ADDED
    logger.info("Smoke test finished: %s", timings)  # ADDED
    return timings  # ADDED


def _finetune_with_fallback(  # ADDED
    config: ExperimentConfig,  # ADDED
    run: RunSpec,  # ADDED
) -> tuple[ExperimentConfig, list[EpochResult]]:  # ADDED
    """Fine-tune; on CUDA out-of-memory retry with batch 4 and 2 accumulation steps."""  # ADDED
    try:  # ADDED
        return config, finetune(config, run)  # ADDED
    except torch.cuda.OutOfMemoryError:  # ADDED
        batch, accum = FALLBACK_BATCH  # ADDED
        logger.warning(  # ADDED
            "Out of memory at batch %d; retrying with %d x %d.",  # ADDED
            config.train.batch_size,  # ADDED
            batch,  # ADDED
            accum,  # ADDED
        )  # ADDED
        torch.cuda.empty_cache()  # ADDED
        smaller = dataclasses.replace(  # ADDED
            config,  # ADDED
            train=dataclasses.replace(config.train, batch_size=batch, grad_accum_steps=accum),  # ADDED
        )  # ADDED
        return smaller, finetune(smaller, run)  # ADDED


def _estimate_full_run(timings: dict[str, Any]) -> dict[str, float]:  # ADDED
    """Hours for the full experiment of ``configs/base.toml`` on this machine.  # ADDED

    Assumes the measured per-iteration and per-batch costs, i.e. the batch size that fitted  # ADDED
    in the smoke test and the same sampling batch size.  # ADDED
    """  # ADDED
    full = load_config()  # ADDED
    train_images = len(load_split(full, "train", limit=full.data.train_limit))  # ADDED
    test_images = len(load_split(full, "test", limit=full.evaluation.test_limit))  # ADDED
    iterations_per_epoch = math.ceil(train_images / timings["train_batch_size"])  # ADDED
    per_batch = timings["sampling_seconds_per_batch"]  # ADDED
    sampling_batches = math.ceil(full.sampling.validation_images / full.sampling.batch_size)  # ADDED
    train_hours = (  # ADDED
        full.train.epochs * iterations_per_epoch * timings["train_seconds_per_iteration"] / 3600  # ADDED
    )  # ADDED
    validation_hours = len(full.train.checkpoint_epochs) * sampling_batches * per_batch / 3600  # ADDED
    test_hours = math.ceil(test_images / full.sampling.batch_size) * per_batch / 3600  # ADDED
    return {  # ADDED
        "train_images": train_images,  # ADDED
        "iterations_per_epoch": iterations_per_epoch,  # ADDED
        "finetune_hours_per_run": round(train_hours, 2),  # ADDED
        "validation_sampling_hours_per_run": round(validation_hours, 2),  # ADDED
        "test_evaluation_hours_per_run": round(test_hours, 2),  # ADDED
        "total_hours_runs_0_A_B": round(2 * (train_hours + validation_hours) + 3 * test_hours, 2),  # ADDED
    }  # ADDED
