# ADDED: new file, not part of the upstream SR3 codebase.
"""Well-known locations inside the repository."""  # ADDED

from pathlib import Path  # ADDED

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ADDED
CONFIG_DIR = PROJECT_ROOT / "configs"  # ADDED
DEFAULT_CONFIG = CONFIG_DIR / "base.toml"  # ADDED
LOCAL_CONFIG = CONFIG_DIR / "local.toml"  # ADDED
UPSTREAM_SR3_DIR = PROJECT_ROOT / "third_party" / "sr3"  # ADDED
FINETUNE_SCRIPT = PROJECT_ROOT / "scripts" / "finetune.py"  # ADDED
REPORT_DIR = PROJECT_ROOT / "report"  # ADDED
