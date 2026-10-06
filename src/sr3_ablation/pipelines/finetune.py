# ADDED: new file, not part of the upstream SR3 codebase.
"""Command: fine-tune the pretrained SR3 model for Run A or Run B."""

from __future__ import annotations

import torch

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.loaders import make_loader
from sr3_ablation.evaluation.metrics import build_metrics
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.model.factory import build_backend
from sr3_ablation.pipelines.common import (
    collect_batches,
    expected_schedule,
    load_split,
    open_run,
    pretrained_weights,
)
from sr3_ablation.runs import RunSpec
from sr3_ablation.sampling.sampler import ReverseSampler
from sr3_ablation.schedule.verification import verify_schedule
from sr3_ablation.training.callbacks import (
    CheckpointSaver,
    LossCsvLogger,
    RunRecordWriter,
    ScheduleEvidenceWriter,
    SnapshotWriter,
    ValidationCallback,
    ValidationMetricLogger,
)
from sr3_ablation.training.trainer import EpochResult, FineTuner

VALIDATION_METRICS = ("psnr", "ssim")


def finetune(config: ExperimentConfig, run: RunSpec) -> list[EpochResult]:
    """Fine-tune from the pretrained weights with the run's schedule and record evidence."""
    if not run.fine_tuned:
        raise ValueError(f"{run.label} is not a fine-tuning run.")
    context = open_run(config, run, "finetune")
    paths, train_cfg, sampling = context.paths, config.train, config.sampling
    weights = pretrained_weights(config)
    backend = build_backend(config.model, config.schedule, run.exponent, context.device, weights)
    schedule = expected_schedule(config, run.exponent)

    train_set = load_split(config, "train", augment=True, limit=config.data.train_limit)
    loader = make_loader(
        train_set,
        train_cfg.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        seed=train_cfg.seed,
    )
    validation_set = load_split(config, "val", limit=sampling.validation_images)
    validation = ValidationCallback(
        ReverseSampler(backend, sampling.seed),
        collect_batches(validation_set, sampling.batch_size),
        train_cfg.checkpoint_epochs,
        sinks=[
            SnapshotWriter(run.name, paths.samples, sampling.snapshot_samples),
            ValidationMetricLogger(
                paths.validation_csv, build_metrics(VALIDATION_METRICS, context.device)
            ),
        ],
        seed=sampling.seed,
    )
    record = {
        "run": run.name,
        "label": run.label,
        "exponent": run.exponent,
        "schedule": schedule.description,
        "pretrained_weights": weights.name,
        "train_images": len(train_set),
        "validation_images": validation_set.image_ids,
        "batch_size": train_cfg.batch_size,
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "effective_batch_size": train_cfg.effective_batch_size,
        "learning_rate": train_cfg.learning_rate,
        "seed": train_cfg.seed,
    }
    trainer = FineTuner(
        backend,
        torch.optim.Adam(backend.parameters(), lr=train_cfg.learning_rate),
        loader,
        epochs=train_cfg.epochs,
        grad_accum_steps=train_cfg.grad_accum_steps,
        max_iterations=train_cfg.max_iterations,
        log_every=train_cfg.log_every,
        callbacks=[
            ScheduleEvidenceWriter(backend, schedule, paths.schedule_evidence),
            LossCsvLogger(paths.loss_csv),
            validation,
            CheckpointSaver(backend, paths, train_cfg.checkpoint_epochs, record),
            RunRecordWriter(paths.run_record, record),
        ],
    )
    return trainer.fit()


def verify_run_schedule(config: ExperimentConfig, run: RunSpec) -> dict:
    """Command ``verify-schedule``: load the pretrained weights, install the run's schedule and
    check the model buffers against the independent computation (no training)."""
    context = open_run(config, run, "verify-schedule")
    backend = build_backend(
        config.model, config.schedule, run.exponent, context.device, pretrained_weights(config)
    )
    evidence = verify_schedule(backend.schedule_buffers(), expected_schedule(config, run.exponent))
    write_json({**evidence, "stage": "verify-schedule"}, context.paths.schedule_evidence)
    return evidence
