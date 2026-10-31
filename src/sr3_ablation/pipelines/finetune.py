# ADDED: new file, not part of the upstream SR3 codebase.
"""Command: fine-tune the pretrained SR3 model for Run A or Run B."""  # ADDED

from __future__ import annotations  # ADDED

import torch  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.data.loaders import make_loader  # ADDED
from sr3_ablation.evaluation.metrics import build_metrics  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.model.factory import build_backend  # ADDED
from sr3_ablation.model.protocols import WeightStore  # ADDED
from sr3_ablation.pipelines.common import (  # ADDED
    collect_batches,  # ADDED
    expected_schedule,  # ADDED
    load_split,  # ADDED
    open_run,  # ADDED
    pretrained_weights,  # ADDED
)  # ADDED
from sr3_ablation.runs import RunSpec  # ADDED
from sr3_ablation.sampling.sampler import ReverseSampler  # ADDED
from sr3_ablation.schedule.verification import verify_schedule  # ADDED
from sr3_ablation.training.callbacks import (  # ADDED
    CheckpointSaver,  # ADDED
    HeartbeatWriter,  # ADDED
    LossCsvLogger,  # ADDED
    RunRecordWriter,  # ADDED
    ScheduleEvidenceWriter,  # ADDED
    SnapshotWriter,  # ADDED
    ValidationCallback,  # ADDED
    ValidationMetricLogger,  # ADDED
)  # ADDED
from sr3_ablation.training.trainer import EpochResult, FineTuner  # ADDED
from sr3_ablation.utils.heartbeat import touch  # ADDED

VALIDATION_METRICS = ("psnr", "ssim")  # ADDED


def _load_training_resume(  # ADDED
    paths: RunPaths,  # ADDED
    backend: WeightStore,  # ADDED
    optimizer: torch.optim.Optimizer,  # ADDED
) -> int:  # ADDED
    """Return the next epoch to run (1-based) after optional ``training_last.pth``."""  # ADDED
    path = paths.checkpoints / "training_last.pth"  # ADDED
    if not path.is_file():  # ADDED
        return 1  # ADDED
    state = torch.load(path, map_location="cpu", weights_only=False)  # ADDED
    backend.load_denoiser_state(state["denoiser"])  # ADDED
    optimizer.load_state_dict(state["optimizer"])  # ADDED
    return int(state["epoch"]) + 1  # ADDED


def finetune(config: ExperimentConfig, run: RunSpec) -> list[EpochResult]:  # ADDED
    """Fine-tune from the pretrained weights with the run's schedule and record evidence."""  # ADDED
    if not run.fine_tuned:  # ADDED
        raise ValueError(f"{run.label} is not a fine-tuning run.")  # ADDED
    context = open_run(config, run, "finetune")  # ADDED
    paths, train_cfg, sampling = context.paths, config.train, config.sampling  # ADDED
    weights = pretrained_weights(config)  # ADDED
    backend = build_backend(config.model, config.schedule, run, context.device, weights)  # ADDED
    schedule = expected_schedule(config, run)  # ADDED

    train_set = load_split(config, "train", augment=True, limit=config.data.train_limit)  # ADDED
    loader = make_loader(  # ADDED
        train_set,  # ADDED
        train_cfg.batch_size,  # ADDED
        shuffle=True,  # ADDED
        num_workers=config.data.num_workers,  # ADDED
        seed=train_cfg.seed,  # ADDED
    )  # ADDED
    validation_set = load_split(config, "val", limit=sampling.validation_images)  # ADDED
    validation = ValidationCallback(  # ADDED
        ReverseSampler(backend, sampling.seed),  # ADDED
        collect_batches(validation_set, sampling.batch_size),  # ADDED
        train_cfg.checkpoint_epochs,  # ADDED
        sinks=[  # ADDED
            SnapshotWriter(run.name, paths.samples, sampling.snapshot_samples),  # ADDED
            ValidationMetricLogger(  # ADDED
                paths.validation_csv,  # ADDED
                build_metrics(VALIDATION_METRICS, context.device),  # ADDED
            ),  # ADDED
        ],  # ADDED
        seed=sampling.seed,  # ADDED
    )  # ADDED
    record = {  # ADDED
        "run": run.name,  # ADDED
        "label": run.label,  # ADDED
        "exponent": run.exponent,  # ADDED
        "schedule": schedule.description,  # ADDED
        "pretrained_weights": weights.name,  # ADDED
        "train_images": len(train_set),  # ADDED
        "validation_images": validation_set.image_ids,  # ADDED
        "batch_size": train_cfg.batch_size,  # ADDED
        "grad_accum_steps": train_cfg.grad_accum_steps,  # ADDED
        "effective_batch_size": train_cfg.effective_batch_size,  # ADDED
        "learning_rate": train_cfg.learning_rate,  # ADDED
        "seed": train_cfg.seed,  # ADDED
        "power_target": run.power_target,  # ADDED
    }  # ADDED
    optimizer = torch.optim.Adam(backend.parameters(), lr=train_cfg.learning_rate)  # ADDED
    start_epoch = _load_training_resume(paths, backend, optimizer)  # ADDED
    heartbeat = paths.root / "heartbeat.txt"  # ADDED
    trainer = FineTuner(  # ADDED
        backend,  # ADDED
        optimizer,  # ADDED
        loader,  # ADDED
        epochs=train_cfg.epochs,  # ADDED
        grad_accum_steps=train_cfg.grad_accum_steps,  # ADDED
        max_iterations=train_cfg.max_iterations,  # ADDED
        log_every=train_cfg.log_every,  # ADDED
        start_epoch=start_epoch,  # ADDED
        callbacks=[  # ADDED
            HeartbeatWriter(heartbeat),  # ADDED
            ScheduleEvidenceWriter(backend, schedule, paths.schedule_evidence),  # ADDED
            LossCsvLogger(paths.loss_csv),  # ADDED
            validation,  # ADDED
            CheckpointSaver(  # ADDED
                backend,  # ADDED
                paths,  # ADDED
                train_cfg.checkpoint_epochs,  # ADDED
                record,  # ADDED
                optimizer=optimizer,  # ADDED
            ),  # ADDED
            RunRecordWriter(paths.run_record, record),  # ADDED
        ],  # ADDED
    )  # ADDED
    touch(heartbeat)  # ADDED
    return trainer.fit()  # ADDED


def verify_run_schedule(config: ExperimentConfig, run: RunSpec) -> dict:  # ADDED
    """Command ``verify-schedule``: load the pretrained weights, install the run's schedule and  # ADDED
    check the model buffers against the independent computation (no training)."""  # ADDED
    context = open_run(config, run, "verify-schedule")  # ADDED
    backend = build_backend(  # ADDED
        config.model,  # ADDED
        config.schedule,  # ADDED
        run,  # ADDED
        context.device,  # ADDED
        pretrained_weights(config),  # ADDED
    )  # ADDED
    evidence = verify_schedule(backend.schedule_buffers(), expected_schedule(config, run))  # ADDED
    write_json({**evidence, "stage": "verify-schedule"}, context.paths.schedule_evidence)  # ADDED
    return evidence  # ADDED
