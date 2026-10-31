# ADDED: new file, not part of the upstream SR3 codebase.
"""Folder layout of one run's outputs (``outputs/<runName>/...``)."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.io.naming import checkpoint_name  # ADDED


@dataclass(frozen=True)  # ADDED
class RunPaths:  # ADDED
    """Every file a run reads or writes, rooted at ``outputs/<runName>``."""  # ADDED

    root: Path  # ADDED

    @classmethod  # ADDED
    def for_run(cls, output_root: Path, run_name: str) -> RunPaths:  # ADDED
        return cls(output_root / run_name)  # ADDED

    @property  # ADDED
    def checkpoints(self) -> Path:  # ADDED
        return self.root / "checkpoints"  # ADDED

    @property  # ADDED
    def samples(self) -> Path:  # ADDED
        return self.root / "samples"  # ADDED

    @property  # ADDED
    def trajectories(self) -> Path:  # ADDED
        return self.root / "trajectories"  # ADDED

    @property  # ADDED
    def test_outputs(self) -> Path:  # ADDED
        return self.root / "test_outputs"  # ADDED

    @property  # ADDED
    def loss_csv(self) -> Path:  # ADDED
        return self.root / "loss.csv"  # ADDED

    @property  # ADDED
    def validation_csv(self) -> Path:  # ADDED
        return self.root / "validation.csv"  # ADDED

    @property  # ADDED
    def metrics_csv(self) -> Path:  # ADDED
        return self.root / "test_metrics.csv"  # ADDED

    @property  # ADDED
    def metrics_summary(self) -> Path:  # ADDED
        return self.root / "test_metrics.json"  # ADDED

    @property  # ADDED
    def schedule_evidence(self) -> Path:  # ADDED
        """Schedule verified inside the model during fine-tuning."""  # ADDED
        return self.root / "schedule.json"  # ADDED

    @property  # ADDED
    def sampling_schedule_evidence(self) -> Path:  # ADDED
        """Schedule verified inside the model when generating test outputs."""  # ADDED
        return self.root / "schedule_sampling.json"  # ADDED

    @property  # ADDED
    def trajectory_record(self) -> Path:  # ADDED
        return self.trajectories / "trajectory.json"  # ADDED

    @property  # ADDED
    def diagnostics(self) -> Path:  # ADDED
        """Analysis-only experiments, kept out of the main results."""  # ADDED
        return self.root / "diagnostics"  # ADDED

    @property  # ADDED
    def start_diagnostic_record(self) -> Path:  # ADDED
        return self.diagnostics / "start_diagnostic.json"  # ADDED

    @property  # ADDED
    def run_record(self) -> Path:  # ADDED
        return self.root / "run.json"  # ADDED

    @property  # ADDED
    def environment(self) -> Path:  # ADDED
        return self.root / "environment.json"  # ADDED

    @property  # ADDED
    def resolved_config(self) -> Path:  # ADDED
        return self.root / "resolved_config.json"  # ADDED

    @property  # ADDED
    def log_file(self) -> Path:  # ADDED
        return self.root / "log.txt"  # ADDED

    def checkpoint(self, epoch: int) -> Path:  # ADDED
        return self.checkpoints / checkpoint_name(epoch)  # ADDED

    def final_checkpoint(self) -> Path:  # ADDED
        return self.checkpoints / "final_gen.pth"  # ADDED

    def create(self) -> RunPaths:  # ADDED
        for folder in (self.root, self.checkpoints, self.samples, self.trajectories):  # ADDED
            folder.mkdir(parents=True, exist_ok=True)  # ADDED
        return self  # ADDED
