# ADDED: new file, not part of the upstream SR3 codebase.
"""Command-line interface: ``python -m sr3_ablation <command> [--config FILE ...]``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from sr3_ablation.config import ExperimentConfig, load_config
from sr3_ablation.paths import CONFIG_DIR, DEFAULT_CONFIG
from sr3_ablation.runs import RunSpec, declared_exponent, run_spec

SMOKE_CONFIGS = [DEFAULT_CONFIG, CONFIG_DIR / "smoke.toml"]

Handler = Callable[[ExperimentConfig, argparse.Namespace], object]


def _run(args: argparse.Namespace) -> RunSpec:
    return run_spec(args.run, declared_exponent())


def _download_data(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.data_prep import download_data

    return download_data(config)


def _download_checkpoint(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.data_prep import download_checkpoint

    return download_checkpoint(config)


def _prepare_data(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.data_prep import prepare_data

    return prepare_data(config, workers=args.workers)


def _verify_schedule(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.finetune import verify_run_schedule

    evidence = verify_run_schedule(config, _run(args))
    return {key: value for key, value in evidence.items() if key != "checks"}


def _run0(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.evaluate import evaluate_run

    return evaluate_run(config, run_spec("0", declared_exponent())).to_dict()


def _finetune(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.finetune import finetune

    return [result.__dict__ for result in finetune(config, _run(args))]


def _evaluate(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.evaluate import evaluate_run

    return evaluate_run(config, _run(args)).to_dict()


def _trajectories(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.evaluate import record_trajectories

    return [path.name for path in record_trajectories(config, _run(args))]


def _report(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.reporting.report import ReportBuilder

    return ReportBuilder(config, declared_exponent()).build(pdf=not args.no_pdf)


def _smoke_test(config: ExperimentConfig, args: argparse.Namespace) -> object:
    from sr3_ablation.pipelines.smoke import smoke_test

    return smoke_test(config, declared_exponent())


COMMANDS: dict[str, tuple[str, Handler]] = {
    "download-data": ("Download and extract AFHQ v1.", _download_data),
    "download-checkpoint": ("Download the pretrained SR3 16->128 generator.", _download_checkpoint),
    "prepare-data": ("Split AFHQ (seeded) and write the SR3 16/128 layout.", _prepare_data),
    "verify-schedule": ("Check the model's schedule buffers for a run.", _verify_schedule),
    "run0": ("Evaluate the pretrained model on the test split (Run 0).", _run0),
    "finetune": ("Fine-tune for Run A or B (same as scripts/finetune.py).", _finetune),
    "evaluate": ("Evaluate a run's final checkpoint on the test split.", _evaluate),
    "trajectories": ("Save x_t at the configured reverse timesteps.", _trajectories),
    "report": ("Build report/build/report.{md,html,pdf}.", _report),
    "smoke-test": ("Short end-to-end run with timing estimates.", _smoke_test),
}
RUN_CHOICES = {"verify-schedule": "AB", "finetune": "AB", "evaluate": "0AB", "trajectories": "0AB"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sr3_ablation",
        description="SR3 noise-schedule ablation (theta_t -> theta_t ** x) on AFHQ.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        action="append",
        type=Path,
        help="TOML file(s), merged in order (default: configs/base.toml; smoke-test default: "
        "base.toml + smoke.toml). configs/local.toml is merged last if present.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (help_text, _) in COMMANDS.items():
        sub = commands.add_parser(name, parents=[common], help=help_text, description=help_text)
        if name in RUN_CHOICES:
            sub.add_argument("--run", required=True, choices=list(RUN_CHOICES[name]))
        if name == "prepare-data":
            sub.add_argument("--workers", type=int, default=8, help="image-resizing threads")
        if name == "report":
            sub.add_argument("--no-pdf", action="store_true", help="only write .md and .html")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_files = args.config or (SMOKE_CONFIGS if args.command == "smoke-test" else None)
    config = load_config(config_files)
    _, handler = COMMANDS[args.command]
    result = handler(config, args)
    print(json.dumps(result, indent=2, default=str))
    return 0
