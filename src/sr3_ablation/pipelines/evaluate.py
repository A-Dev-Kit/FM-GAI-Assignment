# ADDED: new file, not part of the upstream SR3 codebase.
"""Commands: run0 / evaluate (test-set outputs and metrics) and trajectories."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.data.loaders import make_loader  # ADDED
from sr3_ablation.evaluation.evaluator import (  # ADDED
    EvaluationSummary,  # ADDED
    SuperResolutionEvaluator,  # ADDED
    evaluate_bicubic,  # ADDED
)  # ADDED
from sr3_ablation.evaluation.metrics import build_metrics  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.model.factory import build_backend  # ADDED
from sr3_ablation.model.protocols import DiffusionBackend  # ADDED
from sr3_ablation.pipelines.common import (  # ADDED
    RunContext,  # ADDED
    collect_batches,  # ADDED
    expected_schedule,  # ADDED
    load_split,  # ADDED
    open_run,  # ADDED
    weights_for,  # ADDED
)  # ADDED
from sr3_ablation.runs import BICUBIC_RUN, RunSpec  # ADDED
from sr3_ablation.sampling.sampler import ReverseSampler  # ADDED
from sr3_ablation.sampling.trajectory import TrajectoryRecorder, save_trajectory  # ADDED
from sr3_ablation.schedule.verification import verify_schedule  # ADDED


def evaluate_run(config: ExperimentConfig, run: RunSpec) -> EvaluationSummary:  # ADDED
    """Super-resolve the test split with the run's final weights and schedule, then score it.  # ADDED

    For Run 0 the bicubic baseline is scored as well (``outputs/bicubic/``).  # ADDED
    """  # ADDED
    context = open_run(config, run, "evaluate")  # ADDED
    backend = sampling_backend(context)  # ADDED
    test_set = load_split(config, "test", limit=config.evaluation.test_limit)  # ADDED
    loader = make_loader(  # ADDED
        test_set,  # ADDED
        config.sampling.batch_size,  # ADDED
        shuffle=False,  # ADDED
        num_workers=config.data.num_workers,  # ADDED
        seed=config.sampling.seed,  # ADDED
    )  # ADDED
    metrics = build_metrics(config.evaluation.metrics, context.device)  # ADDED
    evaluator = SuperResolutionEvaluator(  # ADDED
        ReverseSampler(backend, config.sampling.seed),  # ADDED
        metrics,  # ADDED
        config.sampling.seed,  # ADDED
    )  # ADDED
    paths = context.paths  # ADDED
    summary = evaluator.evaluate(  # ADDED
        loader,  # ADDED
        run.name,  # ADDED
        run.label,  # ADDED
        paths.test_outputs,  # ADDED
        paths.metrics_csv,  # ADDED
        paths.metrics_summary,  # ADDED
    )  # ADDED
    if not run.fine_tuned:  # ADDED
        bicubic = RunPaths.for_run(config.paths.output_root, BICUBIC_RUN)  # ADDED
        evaluate_bicubic(loader, metrics, bicubic.metrics_csv, bicubic.metrics_summary)  # ADDED
    return summary  # ADDED


def record_trajectories(config: ExperimentConfig, run: RunSpec) -> list[Path]:  # ADDED
    """Save x_t at the configured timesteps for a fixed set of test images."""  # ADDED
    context = open_run(config, run, "trajectories")  # ADDED
    backend = sampling_backend(context)  # ADDED
    sampling = config.sampling  # ADDED
    images = load_split(config, "test", limit=sampling.trajectory_samples)  # ADDED
    (batch,) = collect_batches(images, len(images))  # ADDED
    recorder = TrajectoryRecorder(sampling.trajectory_timesteps)  # ADDED
    ReverseSampler(backend, sampling.seed).sample(batch["SR"], [recorder], seed=sampling.seed)  # ADDED
    written = save_trajectory(recorder.states, run.name, context.paths.trajectories)  # ADDED

    schedule = expected_schedule(config, run)  # ADDED
    write_json(  # ADDED
        {  # ADDED
            "run": run.name,  # ADDED
            "schedule": schedule.description,  # ADDED
            "image_ids": list(batch["image_id"]),  # ADDED
            "states": [  # ADDED
                {  # ADDED
                    "timestep": t,  # ADDED
                    "alpha_bar": schedule.alpha_bar_at(t) if t > 0 else 1.0,  # ADDED
                    "snr": schedule.snr_at(t) if t > 0 else None,  # ADDED
                }  # ADDED
                for t in recorder.states  # ADDED
            ],  # ADDED
        },  # ADDED
        context.paths.trajectory_record,  # ADDED
    )  # ADDED
    return written  # ADDED


def sampling_backend(context: RunContext) -> DiffusionBackend:  # ADDED
    """The run's sampling weights with its schedule installed and verified (evidence saved)."""  # ADDED
    config, run = context.config, context.run  # ADDED
    backend = build_backend(  # ADDED
        config.model,  # ADDED
        config.schedule,  # ADDED
        run,  # ADDED
        context.device,  # ADDED
        weights_for(context),  # ADDED
    )  # ADDED
    evidence = verify_schedule(backend.schedule_buffers(), expected_schedule(config, run))  # ADDED
    write_json({**evidence, "stage": "sampling"}, context.paths.sampling_schedule_evidence)  # ADDED
    return backend  # ADDED
