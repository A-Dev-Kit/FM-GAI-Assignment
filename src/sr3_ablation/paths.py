# ADDED: new file, not part of the upstream SR3 codebase.
"""Well-known locations inside the repository."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "base.toml"
LOCAL_CONFIG = CONFIG_DIR / "local.toml"
UPSTREAM_SR3_DIR = PROJECT_ROOT / "third_party" / "sr3"
FINETUNE_SCRIPT = PROJECT_ROOT / "scripts" / "finetune.py"
REPORT_DIR = PROJECT_ROOT / "report"
