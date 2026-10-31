# ADDED: new file, not part of the upstream SR3 codebase.
"""Identity of the three experimental runs (Run 0, Run A, Run B) and the bicubic reference.  # ADDED

Every run name and label in the project comes from here, so ``runB`` and ``Run B`` are spelled  # ADDED
out in exactly one place.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

import importlib.util  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Literal  # ADDED

from sr3_ablation.paths import FINETUNE_SCRIPT  # ADDED

PowerTarget = Literal["beta", "alpha_bar"]  # ADDED

RUN_KEYS = ("0", "A", "B")  # ADDED
SUPPLEMENTARY_KEY = "S"  # ADDED
FINE_TUNED_KEYS = ("A", "B")  # ADDED
BICUBIC_RUN = "bicubic"  # ADDED
BICUBIC_LABEL = "Bicubic"  # ADDED


def run_name(key: str) -> str:  # ADDED
    """File-system friendly name of a run, e.g. ``runB``."""  # ADDED
    return f"run{key}"  # ADDED


def run_label(key: str) -> str:  # ADDED
    """Human-readable name of a run, e.g. ``Run B``."""  # ADDED
    return f"Run {key}"  # ADDED


ALL_RUNS = {run_name(key): run_label(key) for key in RUN_KEYS}  # ADDED
FINE_TUNED_RUNS = {run_name(key): run_label(key) for key in FINE_TUNED_KEYS}  # ADDED


@dataclass(frozen=True)  # ADDED
class RunSpec:  # ADDED
    """One run of the ablation.  # ADDED

    Attributes:  # ADDED
        key: ``"0"`` (pretrained, no fine-tuning), ``"A"`` (fine-tuned with theta_t) or  # ADDED
            ``"B"`` (fine-tuned and sampled with theta_t ** x).  # ADDED
        exponent: Exponent applied to the noise schedule (1.0 means unchanged).  # ADDED
        fine_tuned: Whether the run fine-tunes the pretrained model.  # ADDED
    """  # ADDED

    key: str  # ADDED
    exponent: float  # ADDED
    fine_tuned: bool  # ADDED
    power_target: PowerTarget = "beta"  # ADDED

    @property  # ADDED
    def name(self) -> str:  # ADDED
        """File-system friendly name used in output paths, e.g. ``runB``."""  # ADDED
        return run_name(self.key)  # ADDED

    @property  # ADDED
    def label(self) -> str:  # ADDED
        """Human-readable name used in figures, e.g. ``Run B``."""  # ADDED
        return run_label(self.key)  # ADDED


def run_spec(key: str, exponent_x: float) -> RunSpec:  # ADDED
    """Build the :class:`RunSpec` for ``key``; ``exponent_x`` is only used by Run B."""  # ADDED
    normalised = key.strip().upper().removeprefix("RUN")  # ADDED
    if normalised == "0":  # ADDED
        return RunSpec(key="0", exponent=1.0, fine_tuned=False)  # ADDED
    if normalised == "A":  # ADDED
        return RunSpec(key="A", exponent=1.0, fine_tuned=True)  # ADDED
    if normalised == "B":  # ADDED
        return RunSpec(key="B", exponent=float(exponent_x), fine_tuned=True, power_target="beta")  # ADDED
    if normalised == "S":  # ADDED
        return RunSpec(  # ADDED
            key="S",  # ADDED
            exponent=float(exponent_x),  # ADDED
            fine_tuned=True,  # ADDED
            power_target="alpha_bar",  # ADDED
        )  # ADDED
    raise ValueError(  # ADDED
        f"Unknown run {key!r}; expected one of {', '.join((*RUN_KEYS, SUPPLEMENTARY_KEY))}."  # ADDED
    )  # ADDED


def declared_exponent(script: Path = FINETUNE_SCRIPT) -> float:  # ADDED
    """Return ``NOISE_EXPONENT_X`` as declared at the top of the fine-tuning script.  # ADDED

    The exponent is declared exactly once (in ``scripts/finetune.py``, as the brief requires);  # ADDED
    every other entry point reads it from there instead of repeating the number.  # ADDED
    """  # ADDED
    spec = importlib.util.spec_from_file_location("sr3_ablation_finetune_script", script)  # ADDED
    if spec is None or spec.loader is None:  # ADDED
        raise FileNotFoundError(f"Fine-tuning script not found: {script}")  # ADDED
    module = importlib.util.module_from_spec(spec)  # ADDED
    spec.loader.exec_module(module)  # ADDED
    return float(module.NOISE_EXPONENT_X)  # ADDED
