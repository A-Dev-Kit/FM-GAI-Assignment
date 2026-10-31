# ADDED: new file, not part of the upstream SR3 codebase.
"""Touch a heartbeat file so a supervisor can detect stalled jobs."""  # ADDED

from __future__ import annotations  # ADDED

import time  # ADDED
from pathlib import Path  # ADDED


def touch(path: Path) -> None:  # ADDED
    """Update ``path`` to the current time."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    path.write_text(str(time.time()), encoding="utf-8")  # ADDED
