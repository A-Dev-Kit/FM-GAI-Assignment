# ADDED: new file, not part of the upstream SR3 codebase.
"""JSON helpers that understand NumPy scalars and paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
