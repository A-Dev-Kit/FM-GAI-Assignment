# ADDED: new file, not part of the upstream SR3 codebase.
"""Section of the inference trajectories: x_t of both runs at the saved timesteps."""

from __future__ import annotations

from sr3_ablation.io.naming import trajectory_name
from sr3_ablation.reporting.grids import image_grid
from sr3_ablation.reporting.sections.base import (
    TRAJECTORIES_BOTH_COMMAND,
    ReportContext,
    pending,
)
from sr3_ablation.runs import FINE_TUNED_RUNS

TRAJECTORY_FIGURE_FILE = "trajectories.png"


class TrajectoryFigureSection:
    """Rows alternate Run A / Run B per sample; columns are the saved timesteps, noisiest first."""

    placeholder = "TRAJECTORY_FIGURE"

    def render(self, context: ReportContext) -> str:
        timesteps = sorted(context.config.sampling.trajectory_timesteps, reverse=True)
        rows, labels = [], []
        for sample in range(1, context.config.sampling.trajectory_samples + 1):
            for name, label in FINE_TUNED_RUNS.items():
                folder = context.run_paths(name).trajectories
                rows.append([folder / trajectory_name(name, t, sample) for t in timesteps])
                labels.append(f"{label}, sample {sample}")
        if not any(path.is_file() for row in rows for path in row):
            return pending(TRAJECTORIES_BOTH_COMMAND)
        path = image_grid(
            rows,
            labels,
            [f"$x_{{{t}}}$" for t in timesteps],
            context.assets / TRAJECTORY_FIGURE_FILE,
            title="Reverse-process states x_t",
        )
        return context.image("Reverse trajectories of Run A and Run B", path)
