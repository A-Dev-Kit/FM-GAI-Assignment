# ADDED: new file, not part of the upstream SR3 codebase.
"""Command: diagnose-start, an analysis-only test of where a run's samples go wrong.  # ADDED

The trajectory images are sampled twice with the same seed: once from pure noise (as in every  # ADDED
reported result) and once from x_T ~ q(x_T | x_0) with the bicubic image as x_0. If a run only  # ADDED
fails from pure noise, the failure comes from the gap between its training distribution at t = T  # ADDED
and the sampler's starting point, not from the denoiser itself.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Sequence  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.evaluation.evaluator import score_images, summarise  # ADDED
from sr3_ablation.evaluation.metrics import ImageMetric, build_metrics  # ADDED
from sr3_ablation.io.images import tensor_to_uint8  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.io.naming import diagnostic_prefix  # ADDED
from sr3_ablation.pipelines.common import collect_batches, expected_schedule, load_split, open_run  # ADDED
from sr3_ablation.pipelines.evaluate import sampling_backend  # ADDED
from sr3_ablation.runs import BICUBIC_RUN, RunSpec  # ADDED
from sr3_ablation.sampling.sampler import ReverseSampler  # ADDED
from sr3_ablation.sampling.starts import PURE_NOISE, ForwardNoisedStart, StartState  # ADDED
from sr3_ablation.sampling.trajectory import TrajectoryRecorder, save_trajectory  # ADDED


def diagnose_start(config: ExperimentConfig, run: RunSpec) -> dict[str, Any]:  # ADDED
    """Sample the trajectory images from both starts; save x_t and the scores of x_0."""  # ADDED
    context = open_run(config, run, "diagnose-start")  # ADDED
    backend = sampling_backend(context)  # ADDED
    sampling = config.sampling  # ADDED
    schedule = expected_schedule(config, run)  # ADDED
    horizon = schedule.n_timestep  # ADDED
    images = load_split(config, "test", limit=sampling.trajectory_samples)  # ADDED
    (batch,) = collect_batches(images, len(images))  # ADDED
    image_ids = list(batch["image_id"])  # ADDED
    metrics = build_metrics(config.evaluation.metrics, context.device)  # ADDED
    targets = tensor_to_uint8(batch["HR"])  # ADDED
    timesteps = sorted({horizon, *sampling.trajectory_timesteps}, reverse=True)  # ADDED
    forward = ForwardNoisedStart(schedule.alpha_bar_at(horizon))  # ADDED

    scores = {BICUBIC_RUN: _scores(tensor_to_uint8(batch["SR"]), targets, image_ids, metrics)}  # ADDED
    starts: tuple[StartState, ...] = (PURE_NOISE, forward)  # ADDED
    for start in starts:  # ADDED
        recorder = TrajectoryRecorder(timesteps)  # ADDED
        sampler = ReverseSampler(backend, sampling.seed, start=start)  # ADDED
        final = sampler.sample(batch["SR"], [recorder], seed=sampling.seed)  # ADDED
        prefix = diagnostic_prefix(run.name, start.name)  # ADDED
        save_trajectory(recorder.states, prefix, context.paths.diagnostics)  # ADDED
        scores[start.name] = _scores(tensor_to_uint8(final), targets, image_ids, metrics)  # ADDED

    record = {  # ADDED
        "run": run.name,  # ADDED
        "schedule": schedule.description,  # ADDED
        "seed": sampling.seed,  # ADDED
        "image_ids": image_ids,  # ADDED
        "timesteps": timesteps,  # ADDED
        "alpha_bar_T": forward.alpha_bar_t,  # ADDED
        "signal_coefficient": forward.signal_coefficient,  # ADDED
        "noise_coefficient": forward.noise_coefficient,  # ADDED
        "starts": [start.name for start in starts],  # ADDED
        "scores": scores,  # ADDED
    }  # ADDED
    write_json(record, context.paths.start_diagnostic_record)  # ADDED
    return {name: entry["mean"] for name, entry in scores.items()}  # ADDED


def _scores(  # ADDED
    predictions: Sequence[np.ndarray],  # ADDED
    targets: Sequence[np.ndarray],  # ADDED
    image_ids: Sequence[str],  # ADDED
    metrics: Sequence[ImageMetric],  # ADDED
) -> dict[str, Any]:  # ADDED
    rows = score_images(predictions, targets, image_ids, metrics)  # ADDED
    return {"per_image": rows, "mean": summarise(rows, metrics, "", "").mean}  # ADDED
