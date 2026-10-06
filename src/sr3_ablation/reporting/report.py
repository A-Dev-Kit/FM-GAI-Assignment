# ADDED: new file, not part of the upstream SR3 codebase.
"""Assemble the report: generate figures and tables from ``outputs/`` and fill the template."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.preprocess import layout_for, processed_root
from sr3_ablation.io.jsonio import read_json
from sr3_ablation.io.naming import epoch_sample_name, super_resolved_name, trajectory_name
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.paths import REPORT_DIR
from sr3_ablation.reporting.grids import image_grid
from sr3_ablation.reporting.markdown_html import markdown_to_html
from sr3_ablation.reporting.pdf import find_browser, html_to_pdf, stamp_page_numbers
from sr3_ablation.reporting.plots import (
    plot_loss_curves,
    plot_metric_curves,
    plot_schedule_comparison,
)
from sr3_ablation.reporting.tables import (
    collect_summaries,
    markdown_table,
    results_table,
    schedule_table,
    snr_at_timesteps_table,
    write_results_csv,
)
from sr3_ablation.schedule.noise_schedule import NoiseSchedule
from sr3_ablation.schedule.transforms import IdentityTransform, PowerTransform
from sr3_ablation.utils.log import get_logger
from sr3_ablation.utils.selection import evenly_spaced_indices

FINE_TUNED_RUNS = {"runA": "Run A", "runB": "Run B"}
COMPARISON_IMAGES = 3
TITLE = "Noise Schedule Ablation in SR3: theta_t vs theta_t^x on AFHQ"
RUNNING_TITLE = "CSL7860 Group Assignment - Noise Schedule Ablation (SR3, AFHQ)"
_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

logger = get_logger("reporting")


def pending(command: str) -> str:
    return f"> *Pending: run `{command}` to generate this item.*"


def fill_template(template: str, values: Mapping[str, str]) -> str:
    """Replace ``{{NAME}}`` placeholders; unknown names raise ``KeyError``."""
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


class ReportBuilder:
    def __init__(
        self,
        config: ExperimentConfig,
        exponent: float,
        template: Path = REPORT_DIR / "report_template.md",
        build_dir: Path = REPORT_DIR / "build",
    ) -> None:
        self._config = config
        self._exponent = exponent
        self._template = template
        self._build_dir = build_dir
        self._outputs = config.paths.output_root
        self._assets = self._outputs / "report_assets"
        self._original = NoiseSchedule.from_config(config.schedule, IdentityTransform())
        self._modified = NoiseSchedule.from_config(config.schedule, PowerTransform(exponent))

    def build(self, pdf: bool = True) -> Path:
        sections: dict[str, Callable[[], str]] = {
            "EXPONENT": lambda: f"{self._exponent:.2f}",
            "SCHEDULE_TABLE": lambda: schedule_table(self._original, self._modified),
            "TRAJECTORY_SNR_TABLE": self._trajectory_snr_table,
            "SCHEDULE_FIGURE": self._schedule_figure,
            "DATASET_TABLE": self._dataset_table,
            "SETUP_TABLE": self._setup_table,
            "ENVIRONMENT": self._environment,
            "SCHEDULE_EVIDENCE": self._schedule_evidence,
            "LOSS_FIGURE": self._loss_figure,
            "LOSS_TABLE": self._loss_table,
            "VALIDATION_FIGURE": self._validation_figure,
            "SNAPSHOT_FIGURES": self._snapshot_figures,
            "RESULTS_TABLE": self._results_table,
            "COMPARISON_FIGURE": self._comparison_figure,
            "TRAJECTORY_FIGURE": self._trajectory_figure,
        }
        text = fill_template(
            self._template.read_text(encoding="utf-8"),
            {name: render() for name, render in sections.items()},
        )
        self._build_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self._build_dir / "report.md"
        markdown_path.write_text(text, encoding="utf-8")
        html_path = markdown_path.with_suffix(".html")
        html_path.write_text(markdown_to_html(text, TITLE), encoding="utf-8")
        logger.info("Wrote %s and %s", markdown_path, html_path)
        if pdf:
            self._print_pdf(html_path)
        return markdown_path

    def _print_pdf(self, html_path: Path) -> None:
        browser = find_browser()
        if browser is None:
            logger.warning("No Edge/Chrome found; open report.html and print it to PDF instead.")
            return
        pdf_path = html_to_pdf(html_path, html_path.with_suffix(".pdf"), browser)
        pages = stamp_page_numbers(pdf_path, RUNNING_TITLE)
        logger.info("Wrote %s (%d pages)", pdf_path, pages)

    # Sections -----------------------------------------------------------------------------

    def _trajectory_snr_table(self) -> str:
        return snr_at_timesteps_table(
            self._original, self._modified, self._config.sampling.trajectory_timesteps
        )

    def _schedule_figure(self) -> str:
        path = plot_schedule_comparison(
            self._config.schedule,
            self._exponent,
            self._config.sampling.trajectory_timesteps,
            self._assets / "schedule_comparison.png",
        )
        return self._image("Original and power-transformed schedules", path)

    def _dataset_table(self) -> str:
        data = self._config.data
        root = processed_root(
            self._config.paths.data_root, data.low_resolution, data.high_resolution
        )
        path = root / "split_summary.json"
        if not path.is_file():
            return pending("python -m sr3_ablation prepare-data")
        counts = read_json(path)["counts"]
        animals = sorted({animal for per_split in counts.values() for animal in per_split})
        rows = [
            [split, *[per_class.get(animal, 0) for animal in animals], sum(per_class.values())]
            for split, per_class in counts.items()
        ]
        return markdown_table(["Split", *animals, "Total"], rows)

    def _setup_table(self) -> str:
        cfg = self._config
        rows = [
            ["Starting checkpoint", f"`{cfg.model.pretrained_file}` (FFHQ, 16→128)"],
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

    def _environment(self) -> str:
        for name in ("runB", "runA", "run0"):
            path = RunPaths.for_run(self._outputs, name).environment
            if path.is_file():
                env = read_json(path)
                gpu = env.get("gpu", {}).get("name", env.get("device"))
                return (
                    f"Python {env['python']}, PyTorch {env['packages'].get('torch')} "
                    f"(CUDA {env.get('torch_cuda')}), {gpu}, {env['platform']}."
                )
        return pending("python -m sr3_ablation run0")

    def _schedule_evidence(self) -> str:
        rows = []
        for name, label in FINE_TUNED_RUNS.items():
            path = RunPaths.for_run(self._outputs, name).schedule_evidence
            if path.is_file():
                evidence = read_json(path)
                rows.append(
                    [
                        label,
                        evidence["description"],
                        f"{evidence['observed_alpha_bar_last']:.6g}",
                        f"{evidence['alpha_bar_last']:.6g}",
                        "yes" if evidence["verified"] else "NO",
                        evidence["stage"],
                    ]
                )
        if not rows:
            return pending("python scripts/finetune.py --run B")
        header = [
            "Run",
            "Schedule",
            "model $\\bar\\alpha_T$",
            "expected $\\bar\\alpha_T$",
            "verified",
            "checked",
        ]
        return markdown_table(header, rows)

    def _loss_series(self) -> dict[str, Path]:
        series = {}
        for name, label in FINE_TUNED_RUNS.items():
            path = RunPaths.for_run(self._outputs, name).loss_csv
            if path.is_file():
                series[label] = path
        return series

    def _loss_figure(self) -> str:
        series = self._loss_series()
        if not series:
            return pending("python scripts/finetune.py --run A  (and --run B)")
        path = plot_loss_curves(series, self._assets / "loss_curves.png")
        return self._image("Training loss per epoch", path)

    def _loss_table(self) -> str:
        series = self._loss_series()
        if not series:
            return ""
        columns = {label: _read_rows(path) for label, path in series.items()}
        epochs = sorted({epoch for rows in columns.values() for epoch in rows})
        body = [[epoch, *[columns[label].get(epoch, "–") for label in columns]] for epoch in epochs]
        return markdown_table(["Epoch", *[f"{label} train_loss" for label in columns]], body)

    def _validation_figure(self) -> str:
        series = {}
        for name, label in FINE_TUNED_RUNS.items():
            path = RunPaths.for_run(self._outputs, name).validation_csv
            if path.is_file():
                series[label] = path
        if not series:
            return pending("python scripts/finetune.py --run A  (and --run B)")
        path = plot_metric_curves(
            series,
            "psnr",
            "PSNR (dB)",
            "Validation PSNR at checkpoint epochs",
            self._assets / "validation_psnr.png",
        )
        return self._image("Validation PSNR at checkpoint epochs", path)

    def _snapshot_figures(self) -> str:
        epochs = self._config.train.checkpoint_epochs
        samples = range(1, self._config.sampling.snapshot_samples + 1)
        parts = []
        for name, label in FINE_TUNED_RUNS.items():
            folder = RunPaths.for_run(self._outputs, name).samples
            rows = [[folder / epoch_sample_name(name, e, s) for e in epochs] for s in samples]
            if not any(path.is_file() for row in rows for path in row):
                continue
            path = image_grid(
                rows,
                [f"sample {s}" for s in samples],
                [f"epoch {e}" for e in epochs],
                self._assets / f"{name}_snapshots.png",
                title=f"{label}: validation samples",
            )
            parts.append(self._image(f"{label} samples at checkpoint epochs", path))
        return "\n\n".join(parts) or pending("python scripts/finetune.py --run A  (and --run B)")

    def _results_table(self) -> str:
        summaries = collect_summaries(self._outputs)
        if not summaries:
            return pending("python -m sr3_ablation run0 / evaluate --run A / evaluate --run B")
        metrics = list(self._config.evaluation.metrics)
        write_results_csv(summaries, metrics, self._assets / "results_table.csv")
        return results_table(summaries, metrics)

    def _comparison_figure(self) -> str:
        cfg = self._config
        layout = layout_for(
            cfg.paths.data_root, "test", cfg.data.low_resolution, cfg.data.high_resolution
        )
        if not layout.hr_dir.is_dir():
            return pending("python -m sr3_ablation prepare-data")
        image_ids = layout.image_ids(cfg.evaluation.test_limit)
        chosen = [image_ids[i] for i in evenly_spaced_indices(len(image_ids), COMPARISON_IMAGES)]
        runs = ("run0", "runA", "runB")
        rows = [
            [layout.lr_path(image_id), layout.sr_path(image_id)]
            + [
                RunPaths.for_run(self._outputs, run).test_outputs
                / super_resolved_name(run, image_id)
                for run in runs
            ]
            + [layout.hr_path(image_id)]
            for image_id in chosen
        ]
        if not any(path.is_file() for row in rows for path in row[2:5]):
            return pending("python -m sr3_ablation run0 / evaluate --run A / evaluate --run B")
        path = image_grid(
            rows,
            chosen,
            ["LR 16×16", "Bicubic", "Run 0", "Run A", "Run B", "HR 128×128"],
            self._assets / "test_comparison.png",
        )
        return self._image("Test images: input, baselines and both fine-tuned runs", path)

    def _trajectory_figure(self) -> str:
        timesteps = sorted(self._config.sampling.trajectory_timesteps, reverse=True)
        rows, labels = [], []
        for sample in range(1, self._config.sampling.trajectory_samples + 1):
            for name, label in FINE_TUNED_RUNS.items():
                folder = RunPaths.for_run(self._outputs, name).trajectories
                rows.append([folder / trajectory_name(name, t, sample) for t in timesteps])
                labels.append(f"{label}, sample {sample}")
        if not any(path.is_file() for row in rows for path in row):
            return pending("python -m sr3_ablation trajectories --run A  (and --run B)")
        path = image_grid(
            rows,
            labels,
            [f"$x_{{{t}}}$" for t in timesteps],
            self._assets / "trajectories.png",
            title="Reverse-process states x_t",
        )
        return self._image("Reverse trajectories of Run A and Run B", path)

    def _image(self, caption: str, path: Path) -> str:
        relative = Path(os.path.relpath(path, self._build_dir)).as_posix()
        return f"![{caption}]({relative})"


def _read_rows(path: Path) -> dict[int, str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
    return {int(epoch): loss for epoch, loss in (line.split(",") for line in lines)}
