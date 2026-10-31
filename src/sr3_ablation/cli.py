# ADDED: new file, not part of the upstream SR3 codebase.
"""Command-line interface: ``python -m sr3_ablation <command> [--config FILE ...]``."""  # ADDED

from __future__ import annotations  # ADDED

import argparse  # ADDED
import json  # ADDED
from collections.abc import Callable, Sequence  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.config import ExperimentConfig, load_config  # ADDED
from sr3_ablation.paths import DEFAULT_CONFIG  # ADDED
from sr3_ablation.runs import RunSpec, declared_exponent, run_spec  # ADDED

Handler = Callable[[ExperimentConfig, argparse.Namespace], object]  # ADDED


def _run(args: argparse.Namespace) -> RunSpec:  # ADDED
    return run_spec(args.run, declared_exponent())  # ADDED


def _download_data(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.data_prep import download_data  # ADDED

    return download_data(config)  # ADDED


def _download_checkpoint(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.data_prep import download_checkpoint  # ADDED

    return download_checkpoint(config)  # ADDED


def _prepare_data(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.data_prep import prepare_data  # ADDED

    return prepare_data(config, workers=args.workers)  # ADDED


def _verify_schedule(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.finetune import verify_run_schedule  # ADDED

    evidence = verify_run_schedule(config, _run(args))  # ADDED
    return {key: value for key, value in evidence.items() if key != "checks"}  # ADDED


def _run0(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.evaluate import evaluate_run  # ADDED

    return evaluate_run(config, run_spec("0", declared_exponent())).to_dict()  # ADDED


def _finetune(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.finetune import finetune  # ADDED

    return [result.__dict__ for result in finetune(config, _run(args))]  # ADDED


def _evaluate(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.evaluate import evaluate_run  # ADDED

    return evaluate_run(config, _run(args)).to_dict()  # ADDED


def _trajectories(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.evaluate import record_trajectories  # ADDED

    return [path.name for path in record_trajectories(config, _run(args))]  # ADDED


def _diagnose_start(config: ExperimentConfig, args: argparse.Namespace) -> object:  # ADDED
    from sr3_ablation.pipelines.diagnostics import diagnose_start  # ADDED

    return diagnose_start(config, _run(args))  # ADDED


COMMANDS: dict[str, tuple[str, Handler]] = {  # ADDED
    "download-data": ("Download and extract Kvasir-SEG.", _download_data),  # ADDED
    "download-checkpoint": ("Download the pretrained SR3 16->128 generator.", _download_checkpoint),  # ADDED
    "prepare-data": (  # ADDED
        "Deduplicate and split Kvasir-SEG (seeded), then write the SR3 16/128 layout.",  # ADDED
        _prepare_data,  # ADDED
    ),  # ADDED
    "verify-schedule": ("Check the model's schedule buffers for a run.", _verify_schedule),  # ADDED
    "run0": ("Evaluate the pretrained model on the test split (Run 0).", _run0),  # ADDED
    "finetune": ("Fine-tune for Run A or B (same as scripts/finetune.py).", _finetune),  # ADDED
    "evaluate": ("Evaluate a run's final checkpoint on the test split.", _evaluate),  # ADDED
    "trajectories": ("Save x_t at the configured reverse timesteps.", _trajectories),  # ADDED
    "diagnose-start": (  # ADDED
        "Analysis only: sample the trajectory images from pure noise and from q(x_T | x_0).",  # ADDED
        _diagnose_start,  # ADDED
    ),  # ADDED
}  # ADDED
RUN_CHOICES = {  # ADDED
    "verify-schedule": "0AB",  # ADDED
    "finetune": "AB",  # ADDED
    "evaluate": "0AB",  # ADDED
    "trajectories": "0AB",  # ADDED
    "diagnose-start": "B",  # ADDED
}  # ADDED


def build_parser() -> argparse.ArgumentParser:  # ADDED
    """Argument parser with one sub-command per entry of ``COMMANDS``."""  # ADDED
    parser = argparse.ArgumentParser(  # ADDED
        prog="python -m sr3_ablation",  # ADDED
        description="SR3 noise-schedule ablation (theta_t -> theta_t ** x) on Kvasir-SEG.",  # ADDED
    )  # ADDED
    common = argparse.ArgumentParser(add_help=False)  # ADDED
    common.add_argument(  # ADDED
        "--config",  # ADDED
        action="append",  # ADDED
        type=Path,  # ADDED
        help="TOML file(s), merged in order (default: configs/base.toml). "  # ADDED
        "configs/local.toml is merged last if present.",  # ADDED
    )  # ADDED
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")  # ADDED
    for name, (help_text, _) in COMMANDS.items():  # ADDED
        sub = commands.add_parser(name, parents=[common], help=help_text, description=help_text)  # ADDED
        if name in RUN_CHOICES:  # ADDED
            sub.add_argument("--run", required=True, choices=list(RUN_CHOICES[name]))  # ADDED
        if name == "prepare-data":  # ADDED
            sub.add_argument("--workers", type=int, default=8, help="image-resizing threads")  # ADDED
    return parser  # ADDED


def main(argv: Sequence[str] | None = None) -> int:  # ADDED
    """Run one command and print its result as JSON; returns the exit code."""  # ADDED
    args = build_parser().parse_args(argv)  # ADDED
    config_files = args.config  # ADDED
    config = load_config(config_files)  # ADDED
    _, handler = COMMANDS[args.command]  # ADDED
    result = handler(config, args)  # ADDED
    print(json.dumps(result, indent=2, default=str))  # ADDED
    return 0  # ADDED
