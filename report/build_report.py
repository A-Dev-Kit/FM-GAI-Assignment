# ADDED: new file, not part of the upstream SR3 codebase.
"""Build report/build/report.{md,html,pdf} from report_template.md and outputs/.

Equivalent to ``python -m sr3_ablation report``. Usage:

    python report/build_report.py [--config configs/base.toml ...] [--no-pdf]
"""

from __future__ import annotations

import sys

from sr3_ablation.cli import main

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))
