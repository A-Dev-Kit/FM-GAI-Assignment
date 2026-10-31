# ADDED: new file, not part of the upstream SR3 codebase.
"""Section of the inference trajectories: x_t of both runs at the saved timesteps."""  # ADDED

from __future__ import annotations  # ADDED

from sr3_ablation.io.naming import trajectory_name  # ADDED
from sr3_ablation.reporting.grids import image_grid  # ADDED
from sr3_ablation.reporting.sections.base import (  # ADDED
    TRAJECTORIES_BOTH_COMMAND,  # ADDED
    ReportContext,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.runs import FINE_TUNED_RUNS  # ADDED

TRAJECTORY_FIGURE_FILE = "trajectories.png"  # ADDED


class TrajectoryFigureSection:  # ADDED
    """Rows alternate Run A / Run B per sample; columns are the saved timesteps, noisiest first."""  # ADDED

    placeholder = "TRAJECTORY_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        timesteps = sorted(context.config.sampling.trajectory_timesteps, reverse=True)  # ADDED
        rows, labels = [], []  # ADDED
        for sample in range(1, context.config.sampling.trajectory_samples + 1):  # ADDED
            for name, label in FINE_TUNED_RUNS.items():  # ADDED
                folder = context.run_paths(name).trajectories  # ADDED
                rows.append([folder / trajectory_name(name, t, sample) for t in timesteps])  # ADDED
                labels.append(f"{label}, sample {sample}")  # ADDED
        if not any(path.is_file() for row in rows for path in row):  # ADDED
            return pending(TRAJECTORIES_BOTH_COMMAND)  # ADDED
        path = image_grid(  # ADDED
            rows,  # ADDED
            labels,  # ADDED
            [f"$x_{{{t}}}$" for t in timesteps],  # ADDED
            context.assets / TRAJECTORY_FIGURE_FILE,  # ADDED
            title="Reverse-process states x_t",  # ADDED
        )  # ADDED
        return context.image("Reverse trajectories of Run A and Run B", path)  # ADDED
