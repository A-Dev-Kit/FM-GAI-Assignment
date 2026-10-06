# ADDED: new file, not part of the upstream SR3 codebase.
"""Folder layout of one run's outputs (``outputs/<runName>/...``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sr3_ablation.io.naming import checkpoint_name


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, output_root: Path, run_name: str) -> RunPaths:
        return cls(output_root / run_name)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def trajectories(self) -> Path:
        return self.root / "trajectories"

    @property
    def test_outputs(self) -> Path:
        return self.root / "test_outputs"

    @property
    def loss_csv(self) -> Path:
        return self.root / "loss.csv"

    @property
    def validation_csv(self) -> Path:
        return self.root / "validation.csv"

    @property
    def metrics_csv(self) -> Path:
        return self.root / "test_metrics.csv"

    @property
    def metrics_summary(self) -> Path:
        return self.root / "test_metrics.json"

    @property
    def schedule_evidence(self) -> Path:
        """Schedule verified inside the model during fine-tuning."""
        return self.root / "schedule.json"

    @property
    def sampling_schedule_evidence(self) -> Path:
        """Schedule verified inside the model when generating test outputs."""
        return self.root / "schedule_sampling.json"

    @property
    def trajectory_record(self) -> Path:
        return self.trajectories / "trajectory.json"

    @property
    def run_record(self) -> Path:
        return self.root / "run.json"

    @property
    def environment(self) -> Path:
        return self.root / "environment.json"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.json"

    @property
    def log_file(self) -> Path:
        return self.root / "log.txt"

    def checkpoint(self, epoch: int) -> Path:
        return self.checkpoints / checkpoint_name(epoch)

    def final_checkpoint(self) -> Path:
        return self.checkpoints / "final_gen.pth"

    def create(self) -> RunPaths:
        for folder in (self.root, self.checkpoints, self.samples, self.trajectories):
            folder.mkdir(parents=True, exist_ok=True)
        return self
