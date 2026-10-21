# ADDED: new file, not part of the upstream SR3 codebase.
"""Command: diagnose-start, an analysis-only test of where a run's samples go wrong.

The trajectory images are sampled twice with the same seed: once from pure noise (as in every
reported result) and once from x_T ~ q(x_T | x_0) with the bicubic image as x_0. If a run only
fails from pure noise, the failure comes from the gap between its training distribution at t = T
and the sampler's starting point, not from the denoiser itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.evaluation.evaluator import score_images, summarise
from sr3_ablation.evaluation.metrics import ImageMetric, build_metrics
from sr3_ablation.io.images import tensor_to_uint8
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.io.naming import diagnostic_prefix
from sr3_ablation.pipelines.common import collect_batches, expected_schedule, load_split, open_run
from sr3_ablation.pipelines.evaluate import sampling_backend
from sr3_ablation.runs import BICUBIC_RUN, RunSpec
from sr3_ablation.sampling.sampler import ReverseSampler
from sr3_ablation.sampling.starts import PURE_NOISE, ForwardNoisedStart, StartState
from sr3_ablation.sampling.trajectory import TrajectoryRecorder, save_trajectory


def diagnose_start(config: ExperimentConfig, run: RunSpec) -> dict[str, Any]:
    """Sample the trajectory images from both starts; save x_t and the scores of x_0."""
    context = open_run(config, run, "diagnose-start")
    backend = sampling_backend(context)
    sampling = config.sampling
    schedule = expected_schedule(config, run.exponent)
    horizon = schedule.n_timestep
    images = load_split(config, "test", limit=sampling.trajectory_samples)
    (batch,) = collect_batches(images, len(images))
    image_ids = list(batch["image_id"])
    metrics = build_metrics(config.evaluation.metrics, context.device)
    targets = tensor_to_uint8(batch["HR"])
    timesteps = sorted({horizon, *sampling.trajectory_timesteps}, reverse=True)
    forward = ForwardNoisedStart(schedule.alpha_bar_at(horizon))

    scores = {BICUBIC_RUN: _scores(tensor_to_uint8(batch["SR"]), targets, image_ids, metrics)}
    starts: tuple[StartState, ...] = (PURE_NOISE, forward)
    for start in starts:
        recorder = TrajectoryRecorder(timesteps)
        sampler = ReverseSampler(backend, sampling.seed, start=start)
        final = sampler.sample(batch["SR"], [recorder], seed=sampling.seed)
        prefix = diagnostic_prefix(run.name, start.name)
        save_trajectory(recorder.states, prefix, context.paths.diagnostics)
        scores[start.name] = _scores(tensor_to_uint8(final), targets, image_ids, metrics)

    record = {
        "run": run.name,
        "schedule": schedule.description,
        "seed": sampling.seed,
        "image_ids": image_ids,
        "timesteps": timesteps,
        "alpha_bar_T": forward.alpha_bar_t,
        "signal_coefficient": forward.signal_coefficient,
        "noise_coefficient": forward.noise_coefficient,
        "starts": [start.name for start in starts],
        "scores": scores,
    }
    write_json(record, context.paths.start_diagnostic_record)
    return {name: entry["mean"] for name, entry in scores.items()}


def _scores(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    image_ids: Sequence[str],
    metrics: Sequence[ImageMetric],
) -> dict[str, Any]:
    rows = score_images(predictions, targets, image_ids, metrics)
    return {"per_image": rows, "mean": summarise(rows, metrics, "", "").mean}
