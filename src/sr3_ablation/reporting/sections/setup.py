# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Experimental Setup: hyperparameter table and software environment."""  # ADDED

from __future__ import annotations  # ADDED

from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.reporting.sections.base import RUN0_COMMAND, ReportContext, pending  # ADDED
from sr3_ablation.reporting.tables import markdown_table  # ADDED
from sr3_ablation.runs import ALL_RUNS  # ADDED

ENVIRONMENT_SOURCES = tuple(reversed(ALL_RUNS))  # ADDED
"""Runs to read ``environment.json`` from, the most recently executed one first."""  # ADDED


class SetupTableSection:  # ADDED
    """Settings shared by Run A and Run B."""  # ADDED

    placeholder = "SETUP_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        cfg = context.config  # ADDED
        rows = [  # ADDED
            ["Starting checkpoint", f"`{cfg.model.pretrained_file}` (FFHQ, 16→128)"],  # ADDED
            [  # ADDED
                "Dataset",  # ADDED
                f"Kvasir-SEG, seeded split: {cfg.data.val_size} validation, "  # ADDED
                f"{cfg.data.test_size} test, rest training",  # ADDED
            ],  # ADDED
            [  # ADDED
                "Resolution",  # ADDED
                f"{cfg.data.low_resolution}×{cfg.data.low_resolution} → "  # ADDED
                f"{cfg.data.high_resolution}×{cfg.data.high_resolution}",  # ADDED
            ],  # ADDED
            ["Diffusion steps $T$", cfg.schedule.n_timestep],  # ADDED
            [  # ADDED
                "Base schedule",  # ADDED
                f"{cfg.schedule.schedule}, $\\beta_1={cfg.schedule.linear_start:g}$, "  # ADDED
                f"$\\beta_T={cfg.schedule.linear_end:g}$",  # ADDED
            ],  # ADDED
            ["Optimiser", f"Adam, learning rate {cfg.train.learning_rate:g}"],  # ADDED
            [  # ADDED
                "Batch size",  # ADDED
                f"{cfg.train.batch_size} × {cfg.train.grad_accum_steps} accumulation "  # ADDED
                f"= {cfg.train.effective_batch_size}",  # ADDED
            ],  # ADDED
            ["Epochs", cfg.train.epochs],  # ADDED
            ["Checkpoint / snapshot epochs", ", ".join(map(str, cfg.train.checkpoint_epochs))],  # ADDED
            ["Loss", "L1 between true and predicted noise (upstream SR3)"],  # ADDED
            ["Augmentation", "random horizontal flip"],  # ADDED
            [  # ADDED
                "Seeds",  # ADDED
                f"split {cfg.data.split_seed}, training {cfg.train.seed}, "  # ADDED
                f"sampling {cfg.sampling.seed}",  # ADDED
            ],  # ADDED
            ["Test metrics", ", ".join(m.upper() for m in cfg.evaluation.metrics)],  # ADDED
        ]  # ADDED
        return markdown_table(["Setting", "Value (identical for Run A and Run B)"], rows)  # ADDED


class EnvironmentSection:  # ADDED
    """Python, PyTorch, CUDA and GPU of the first run that recorded its environment."""  # ADDED

    placeholder = "ENVIRONMENT"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        for name in ENVIRONMENT_SOURCES:  # ADDED
            path = context.run_paths(name).environment  # ADDED
            if path.is_file():  # ADDED
                env = read_json(path)  # ADDED
                gpu = env.get("gpu", {}).get("name", env.get("device"))  # ADDED
                return (  # ADDED
                    f"Python {env['python']}, PyTorch {env['packages'].get('torch')} "  # ADDED
                    f"(CUDA {env.get('torch_cuda')}), {gpu}, {env['platform']}."  # ADDED
                )  # ADDED
        return pending(RUN0_COMMAND)  # ADDED
