# ADDED: new file, not part of the upstream SR3 codebase.
"""Run the full experiment pipeline with stage markers, retries and a heartbeat watchdog."""  # ADDED

from __future__ import annotations  # ADDED

import hashlib  # ADDED
import json  # ADDED
import subprocess  # ADDED
import sys  # ADDED
import time  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.config import ExperimentConfig, load_config  # ADDED
from sr3_ablation.config.loader import save_resolved_config  # ADDED
from sr3_ablation.paths import DEFAULT_CONFIG, PROJECT_ROOT  # ADDED
from sr3_ablation.utils.heartbeat import touch  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

logger = get_logger("supervise")  # ADDED

STALE_SECONDS = 600  # ADDED
MAX_ATTEMPTS = 3  # ADDED
MIN_FREE_GB = 15  # ADDED


def preflight(config: ExperimentConfig) -> None:  # ADDED
    """Fail fast when GPU, disk or optional LPIPS weights are not ready."""  # ADDED
    import shutil  # ADDED

    free_gb = shutil.disk_usage(config.paths.output_root).free / (1024**3)  # ADDED
    if free_gb < MIN_FREE_GB:  # ADDED
        raise RuntimeError(f"Need at least {MIN_FREE_GB} GB free under outputs; found {free_gb:.1f} GB.")  # ADDED
    try:  # ADDED
        import torch  # ADDED

        if torch.cuda.is_available():  # ADDED
            free, total = torch.cuda.mem_get_info()  # ADDED
            if free < 2 * 1024**3:  # ADDED
                raise RuntimeError(  # ADDED
                    f"CUDA memory low: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB."  # ADDED
                )  # ADDED
    except ImportError:  # ADDED
        pass  # ADDED
    if "lpips" in config.evaluation.metrics:  # ADDED
        try:  # ADDED
            import lpips  # noqa: F401  # ADDED
        except ImportError as error:  # ADDED
            raise RuntimeError("Install LPIPS before the supervised run: pip install -e .[lpips]") from error  # ADDED


@dataclass(frozen=True)  # ADDED
class Stage:  # ADDED
    """One supervised subprocess with a done-marker file."""  # ADDED

    name: str  # ADDED
    command: list[str]  # ADDED
    done_marker: Path  # ADDED


def _config_hash(config: ExperimentConfig) -> str:  # ADDED
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)  # ADDED
    return hashlib.sha256(payload.encode()).hexdigest()[:16]  # ADDED


def _marker_dir(config: ExperimentConfig) -> Path:  # ADDED
    return config.paths.output_root / "supervisor"  # ADDED


def _stages(config_path: Path, python: str) -> list[Stage]:  # ADDED
    config = load_config([config_path.resolve()])  # ADDED
    markers = _marker_dir(config)  # ADDED
    cfg = str(config_path.resolve())  # ADDED

    return [  # ADDED
        Stage("prepare-data", [python, "-m", "sr3_ablation", "prepare-data", "--config", cfg], markers / "prepare-data.done"),  # ADDED
        Stage("run0", [python, "-m", "sr3_ablation", "run0", "--config", cfg], markers / "run0.done"),  # ADDED
        Stage("runA_finetune", [python, "-m", "sr3_ablation", "finetune", "--run", "A", "--config", cfg], markers / "runA_finetune.done"),  # ADDED
        Stage("runA_eval", [python, "-m", "sr3_ablation", "evaluate", "--run", "A", "--config", cfg], markers / "runA_eval.done"),  # ADDED
        Stage(  # ADDED
            "runA_trajectories",  # ADDED
            [python, "-m", "sr3_ablation", "trajectories", "--run", "A", "--config", cfg],  # ADDED
            markers / "runA_traj.done",  # ADDED
        ),  # ADDED
        Stage("runB_finetune", [python, "-m", "sr3_ablation", "finetune", "--run", "B", "--config", cfg], markers / "runB_finetune.done"),  # ADDED
        Stage("runB_eval", [python, "-m", "sr3_ablation", "evaluate", "--run", "B", "--config", cfg], markers / "runB_eval.done"),  # ADDED
        Stage(  # ADDED
            "runB_trajectories",  # ADDED
            [python, "-m", "sr3_ablation", "trajectories", "--run", "B", "--config", cfg],  # ADDED
            markers / "runB_trajectories.done",  # ADDED
        ),  # ADDED
        Stage(  # ADDED
            "runB_diagnostic",  # ADDED
            [python, "-m", "sr3_ablation", "diagnose-start", "--run", "B", "--config", cfg],  # ADDED
            markers / "runB_diagnostic.done",  # ADDED
        ),  # ADDED
        Stage("report", [python, "-m", "sr3_ablation", "report", "--config", cfg], markers / "report.done"),  # ADDED
        Stage("package", [python, "-m", "sr3_ablation", "package", "--config", cfg], markers / "package.done"),  # ADDED
        Stage("runS_finetune", [python, "-m", "sr3_ablation", "finetune", "--run", "S", "--config", cfg], markers / "runS_finetune.done"),  # ADDED
        Stage("runS_eval", [python, "-m", "sr3_ablation", "evaluate", "--run", "S", "--config", cfg], markers / "runS_eval.done"),  # ADDED
        Stage("final_report", [python, "-m", "sr3_ablation", "report", "--config", cfg], markers / "final_report.done"),  # ADDED
        Stage("final_package", [python, "-m", "sr3_ablation", "package", "--config", cfg], markers / "final_package.done"),  # ADDED
    ]  # ADDED


def _write_status(path: Path, stage: str, attempt: int, message: str) -> None:  # ADDED
    path.write_text(  # ADDED
        json.dumps(  # ADDED
            {"stage": stage, "attempt": attempt, "message": message, "time": time.time()},  # ADDED
            indent=2,  # ADDED
        ),  # ADDED
        encoding="utf-8",  # ADDED
    )  # ADDED


def _stage_complete(marker: Path, config_hash: str) -> bool:  # ADDED
    if not marker.is_file():  # ADDED
        return False  # ADDED
    return marker.read_text(encoding="utf-8").strip() == config_hash  # ADDED


def _mark_done(marker: Path, config_hash: str) -> None:  # ADDED
    marker.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    marker.write_text(config_hash, encoding="utf-8")  # ADDED


def _run_stage(stage: Stage, cwd: Path, heartbeat: Path, log: Path) -> None:  # ADDED
    touch(heartbeat)  # ADDED
    with log.open("a", encoding="utf-8") as handle:  # ADDED
        handle.write(f"\n=== {stage.name} ===\n")  # ADDED
        handle.flush()  # ADDED
        process = subprocess.Popen(  # ADDED
            stage.command,  # ADDED
            cwd=cwd,  # ADDED
            stdout=handle,  # ADDED
            stderr=subprocess.STDOUT,  # ADDED
        )  # ADDED
        while process.poll() is None:  # ADDED
            touch(heartbeat)  # ADDED
            time.sleep(30)  # ADDED
        if process.returncode != 0:  # ADDED
            raise RuntimeError(f"{stage.name} exited with code {process.returncode}")  # ADDED


def supervise(config_path: Path = DEFAULT_CONFIG, *, python: str | None = None) -> dict[str, str]:  # ADDED
    """Execute every pipeline stage, skipping those already marked done for this config hash."""  # ADDED
    config = load_config([config_path.resolve()])  # ADDED
    preflight(config)  # ADDED
    config_hash = _config_hash(config)  # ADDED
    markers = _marker_dir(config)  # ADDED
    markers.mkdir(parents=True, exist_ok=True)  # ADDED
    status_path = markers / "status.json"  # ADDED
    log_path = markers / "supervisor.log"  # ADDED
    heartbeat = markers / "heartbeat.txt"  # ADDED
    save_resolved_config(config, markers / "resolved_config.json")  # ADDED
    interpreter = python or sys.executable  # ADDED
    cwd = PROJECT_ROOT  # ADDED
    stages = _stages(config_path.resolve(), interpreter)  # ADDED
    completed: list[str] = []  # ADDED
    for stage in stages:  # ADDED
        if _stage_complete(stage.done_marker, config_hash):  # ADDED
            logger.info("Skipping completed stage %s", stage.name)  # ADDED
            completed.append(stage.name)  # ADDED
            continue  # ADDED
        for attempt in range(1, MAX_ATTEMPTS + 1):  # ADDED
            _write_status(status_path, stage.name, attempt, "running")  # ADDED
            logger.info("Stage %s attempt %d", stage.name, attempt)  # ADDED
            try:  # ADDED
                _run_stage(stage, cwd, heartbeat, log_path)  # ADDED
            except RuntimeError as error:  # ADDED
                logger.warning("%s", error)  # ADDED
                if attempt == MAX_ATTEMPTS:  # ADDED
                    _write_status(status_path, stage.name, attempt, str(error))  # ADDED
                    raise  # ADDED
                time.sleep(60 * attempt)  # ADDED
                continue  # ADDED
            _mark_done(stage.done_marker, config_hash)  # ADDED
            completed.append(stage.name)  # ADDED
            break  # ADDED
    _write_status(status_path, "done", 0, "all stages finished")  # ADDED
    return {"config_hash": config_hash, "completed": ",".join(completed)}  # ADDED
