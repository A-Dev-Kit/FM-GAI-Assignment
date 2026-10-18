# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Experimental Setup: hyperparameter table and software environment."""

from __future__ import annotations

from sr3_ablation.io.jsonio import read_json
from sr3_ablation.reporting.sections.base import RUN0_COMMAND, ReportContext, pending
from sr3_ablation.reporting.tables import markdown_table
from sr3_ablation.runs import ALL_RUNS

ENVIRONMENT_SOURCES = tuple(reversed(ALL_RUNS))
"""Runs to read ``environment.json`` from, the most recently executed one first."""


class SetupTableSection:
    """Settings shared by Run A and Run B."""

    placeholder = "SETUP_TABLE"

    def render(self, context: ReportContext) -> str:
        cfg = context.config
        rows = [
            ["Starting checkpoint", f"`{cfg.model.pretrained_file}` (FFHQ, 16→128)"],
            [
                "Dataset",
                f"Kvasir-SEG, seeded split: {cfg.data.val_size} validation, "
                f"{cfg.data.test_size} test, rest training",
            ],
            [
                "Resolution",
                f"{cfg.data.low_resolution}×{cfg.data.low_resolution} → "
                f"{cfg.data.high_resolution}×{cfg.data.high_resolution}",
            ],
            ["Diffusion steps $T$", cfg.schedule.n_timestep],
            [
                "Base schedule",
                f"{cfg.schedule.schedule}, $\\beta_1={cfg.schedule.linear_start:g}$, "
                f"$\\beta_T={cfg.schedule.linear_end:g}$",
            ],
            ["Optimiser", f"Adam, learning rate {cfg.train.learning_rate:g}"],
            [
                "Batch size",
                f"{cfg.train.batch_size} × {cfg.train.grad_accum_steps} accumulation "
                f"= {cfg.train.effective_batch_size}",
            ],
            ["Epochs", cfg.train.epochs],
            ["Checkpoint / snapshot epochs", ", ".join(map(str, cfg.train.checkpoint_epochs))],
            ["Loss", "L1 between true and predicted noise (upstream SR3)"],
            ["Augmentation", "random horizontal flip"],
            [
                "Seeds",
                f"split {cfg.data.split_seed}, training {cfg.train.seed}, "
                f"sampling {cfg.sampling.seed}",
            ],
            ["Test metrics", ", ".join(m.upper() for m in cfg.evaluation.metrics)],
        ]
        return markdown_table(["Setting", "Value (identical for Run A and Run B)"], rows)


class EnvironmentSection:
    """Python, PyTorch, CUDA and GPU of the first run that recorded its environment."""

    placeholder = "ENVIRONMENT"

    def render(self, context: ReportContext) -> str:
        for name in ENVIRONMENT_SOURCES:
            path = context.run_paths(name).environment
            if path.is_file():
                env = read_json(path)
                gpu = env.get("gpu", {}).get("name", env.get("device"))
                return (
                    f"Python {env['python']}, PyTorch {env['packages'].get('torch')} "
                    f"(CUDA {env.get('torch_cuda')}), {gpu}, {env['platform']}."
                )
        return pending(RUN0_COMMAND)
