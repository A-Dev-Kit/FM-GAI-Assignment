# ADDED: new file, not part of the upstream SR3 codebase.
"""Identity of the three experimental runs (Run 0, Run A, Run B)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from sr3_ablation.paths import FINETUNE_SCRIPT

RUN_KEYS = ("0", "A", "B")


@dataclass(frozen=True)
class RunSpec:
    """One run of the ablation.

    Attributes:
        key: ``"0"`` (pretrained, no fine-tuning), ``"A"`` (fine-tuned with theta_t) or
            ``"B"`` (fine-tuned and sampled with theta_t ** x).
        exponent: Exponent applied to the noise schedule (1.0 means unchanged).
        fine_tuned: Whether the run fine-tunes the pretrained model.
    """

    key: str
    exponent: float
    fine_tuned: bool

    @property
    def name(self) -> str:
        """File-system friendly name used in output paths, e.g. ``runB``."""
        return f"run{self.key}"

    @property
    def label(self) -> str:
        """Human-readable name used in figures, e.g. ``Run B``."""
        return f"Run {self.key}"


def run_spec(key: str, exponent_x: float) -> RunSpec:
    """Build the :class:`RunSpec` for ``key``; ``exponent_x`` is only used by Run B."""
    normalised = key.strip().upper().removeprefix("RUN")
    if normalised == "0":
        return RunSpec(key="0", exponent=1.0, fine_tuned=False)
    if normalised == "A":
        return RunSpec(key="A", exponent=1.0, fine_tuned=True)
    if normalised == "B":
        return RunSpec(key="B", exponent=float(exponent_x), fine_tuned=True)
    raise ValueError(f"Unknown run {key!r}; expected one of {', '.join(RUN_KEYS)}.")


def declared_exponent(script: Path = FINETUNE_SCRIPT) -> float:
    """Return ``NOISE_EXPONENT_X`` as declared at the top of the fine-tuning script.

    The exponent is declared exactly once (in ``scripts/finetune.py``, as the brief requires);
    every other entry point reads it from there instead of repeating the number.
    """
    spec = importlib.util.spec_from_file_location("sr3_ablation_finetune_script", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Fine-tuning script not found: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return float(module.NOISE_EXPONENT_X)
