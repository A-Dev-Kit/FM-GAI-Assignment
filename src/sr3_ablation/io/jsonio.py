# ADDED: new file, not part of the upstream SR3 codebase.
"""JSON helpers that understand NumPy scalars and paths."""  # ADDED

from __future__ import annotations  # ADDED

import json  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED


def write_json(data: Any, path: Path) -> None:  # ADDED
    """Write ``data`` as indented JSON; NumPy values and paths are converted."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    path.write_text(json.dumps(data, indent=2, default=_default) + "\n", encoding="utf-8")  # ADDED


def read_json(path: Path) -> Any:  # ADDED
    """Parse a UTF-8 JSON file."""  # ADDED
    return json.loads(path.read_text(encoding="utf-8"))  # ADDED


def _default(value: Any) -> Any:  # ADDED
    if isinstance(value, np.generic):  # ADDED
        return value.item()  # ADDED
    if isinstance(value, np.ndarray):  # ADDED
        return value.tolist()  # ADDED
    if isinstance(value, Path):  # ADDED
        return value.as_posix()  # ADDED
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")  # ADDED
