# ADDED: new file, not part of the upstream SR3 codebase.
"""Commands: run0 / evaluate (test-set outputs and metrics) and trajectories."""

from __future__ import annotations

from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.loaders import make_loader
from sr3_ablation.evaluation.evaluator import (
    EvaluationSummary,
    SuperResolutionEvaluator,
    evaluate_bicubic,
)
from sr3_ablation.evaluation.metrics import build_metrics
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.model.factory import build_backend
from sr3_ablation.model.protocols import DiffusionBackend
from sr3_ablation.pipelines.common import (
    RunContext,
    collect_batches,
    expected_schedule,
    load_split,
    open_run,
    weights_for,
)
from sr3_ablation.runs import RunSpec
from sr3_ablation.sampling.sampler import ReverseSampler
from sr3_ablation.sampling.trajectory import TrajectoryRecorder, save_trajectory
from sr3_ablation.schedule.verification import verify_schedule

BICUBIC_RUN = "bicubic"


def evaluate_run(config: ExperimentConfig, run: RunSpec) -> EvaluationSummary:
    """Super-resolve the test split with the run's final weights and schedule, then score it.

    For Run 0 the bicubic baseline is scored as well (``outputs/bicubic/``).
    """
    context = open_run(config, run, "evaluate")
    backend = _sampling_backend(context)
    test_set = load_split(config, "test", limit=config.evaluation.test_limit)
    loader = make_loader(
        test_set,
        config.sampling.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        seed=config.sampling.seed,
    )
    metrics = build_metrics(config.evaluation.metrics, context.device)
    evaluator = SuperResolutionEvaluator(
        ReverseSampler(backend, config.sampling.seed), metrics, config.sampling.seed
    )
    paths = context.paths
    summary = evaluator.evaluate(
        loader, run.name, run.label, paths.test_outputs, paths.metrics_csv, paths.metrics_summary
    )
    if not run.fine_tuned:
        bicubic = RunPaths.for_run(config.paths.output_root, BICUBIC_RUN)
        evaluate_bicubic(loader, metrics, bicubic.metrics_csv, bicubic.metrics_summary)
    return summary


def record_trajectories(config: ExperimentConfig, run: RunSpec) -> list[Path]:
    """Save x_t at the configured timesteps for a fixed set of test images."""
    context = open_run(config, run, "trajectories")
    backend = _sampling_backend(context)
    sampling = config.sampling
    images = load_split(config, "test", limit=sampling.trajectory_samples)
    (batch,) = collect_batches(images, len(images))
    recorder = TrajectoryRecorder(sampling.trajectory_timesteps)
    ReverseSampler(backend, sampling.seed).sample(batch["SR"], [recorder], seed=sampling.seed)
    written = save_trajectory(recorder.states, run.name, context.paths.trajectories)

    schedule = expected_schedule(config, run.exponent)
    write_json(
        {
            "run": run.name,
            "schedule": schedule.description,
            "image_ids": list(batch["image_id"]),
            "states": [
                {
                    "timestep": t,
                    "alpha_bar": schedule.alpha_bar_at(t) if t > 0 else 1.0,
                    "snr": schedule.snr_at(t) if t > 0 else None,
                }
                for t in recorder.states
            ],
        },
        context.paths.trajectory_record,
    )
    return written


def _sampling_backend(context: RunContext) -> DiffusionBackend:
    config, run = context.config, context.run
    backend = build_backend(
        config.model, config.schedule, run.exponent, context.device, weights_for(context)
    )
    evidence = verify_schedule(backend.schedule_buffers(), expected_schedule(config, run.exponent))
    write_json({**evidence, "stage": "sampling"}, context.paths.sampling_schedule_evidence)
    return backend
